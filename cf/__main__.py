from json import dumps
from cf import resource_filepath, CollaborativeFiltering
from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser(description="Collaborative filtering runner")
    parser.add_argument("--user-id", type=int, help="User id", dest="user_id", default=1)
    parser.add_argument("--ranks-filename", type=str, help="Ranks filename", dest="ranks_filename", default="u.base")
    parser.add_argument("--items-filename", type=str, help="Items filename", dest="items_filename", default="u.item")
    parser.add_argument("--model-filename", type=str, help="Model filename", dest="model_filename", default="model.pkl")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--test", action="store_true")
    return parser.parse_args()


def train(model_filepath, items_filepath, ranks_filepath):
    items_columns = ["movie id", "movie title", "release date", "video release date", "IMDb URL", "unknown", "Action",
                     "Adventure",
                     "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                     "Horror",
                     "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    ranks_columns = ["user_id", "item_id", "rating", "timestamp"]
    cf = CollaborativeFiltering()
    cf.load_data(items_filepath=items_filepath, ranks_filepath=ranks_filepath, items_columns=items_columns,
                 ranks_columns=ranks_columns)

    cf.train()
    cf.serialize(model_filepath)
    print(dumps({"model_filepath": model_filepath}, indent=4))


def test(model_filepath):
    cf = CollaborativeFiltering.deserialize(model_filepath)
    predictions = cf.predict(args.user_id)
    print(dumps({"predictions": predictions, "user-id": args.user_id}, indent=4))



def main(args):
    model_filepath = resource_filepath(args.model_filename)
    if args.train:
        items_filepath = resource_filepath(args.items_filename)
        ranks_filepath = resource_filepath(args.ranks_filename)
        train(model_filepath, items_filepath, ranks_filepath)
    elif args.test:
        test(model_filepath)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
