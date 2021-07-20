import pandas as pd
from scipy.spatial import distance
from pickle import dump, load


class CollaborativeFiltering:
    def load_data(self, items_filepath, ranks_filepath, items_columns, ranks_columns):
        self.items = pd.read_csv(items_filepath, sep='|', header=None, names = items_columns)
        self.ranks = pd.read_csv(ranks_filepath, sep='\t', header=None, names = ranks_columns)

    def train(self):
        cf_data = self.ranks.pivot(columns="item_id", index="user_id", values="rating").reindex(labels=range(1, 11), columns=self.items["movie id"])
        self.cf_data_final = cf_data.fillna(3.528)
        self.cf_data_final_numpy = self.cf_data_final.to_numpy()

    def predict(self, user_id):
        # results = []
        # for id, row in self.items.iterrows():
        #     results.append(self._get_avg_rank(user_id, item=id))
        # return results
        return [self._get_avg_rank(user_id, item) for item in self.items["movie id"]]

    def _get_avg_rank(self, user_id, item):
        def compute_rank(array, user):
            rank = {}
            for id, row in enumerate(array):
                dist = 1 - distance.cosine(user, row)
                rank[dist] = id
            return rank

        cf_rank = compute_rank(self.cf_data_final_numpy, self.cf_data_final.iloc[user_id].to_numpy())
        neighbours = sorted(cf_rank.items(), key=lambda x: x[0], reverse=True)[1:3]
        neighbours_ids = [value for _, value in neighbours]

        ranks = self.cf_data_final[item].iloc[neighbours_ids]
        #     print(ranks)
        #     print(ranks.mean())
        return ranks.mean()

    def serialize(self, model_filepath):
        with open(model_filepath, "wb") as f:
            dump(self, f)

    @staticmethod
    def deserialize(model_filepath):
        with open(model_filepath, "rb") as f:
            return load(f)
