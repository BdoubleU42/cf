from pytest import fixture
from cf.utils import resource_filepath, remove_file
from cf.cf import CollaborativeFiltering


@fixture
def cf():
    items_columns = ["movie id", "movie title", "release date", "video release date", "IMDb URL", "unknown", "Action",
                     "Adventure",
                     "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                     "Horror",
                     "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    ranks_columns = ["user_id", "item_id", "rating", "timestamp"]
    items_filepath = resource_filepath("u.item")
    ranks_filepath = resource_filepath("u.base")
    cf = CollaborativeFiltering()
    cf.load_data(items_filepath=items_filepath, ranks_filepath=ranks_filepath, items_columns=items_columns,
                 ranks_columns=ranks_columns)

    return cf


@fixture(scope="session")
def model_filepath():
    model_filepath = resource_filepath("model.pkl")
    yield model_filepath
    remove_file(model_filepath)

def test_cf(cf):
    cf.train()
    predictions = cf.predict(user_id=2)
    print(predictions)

    assert isinstance(predictions, list)


def test_serialization(cf, model_filepath):
    cf.train()
    cf.serialize(model_filepath)
    cf_deserialized = CollaborativeFiltering.deserialize(model_filepath)

    assert (cf_deserialized.cf_data_final_numpy == cf.cf_data_final_numpy).all()
