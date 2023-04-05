from research.relevance_statistics import save_model_variant_relevances
from research.user_based.user_relevance_eval import user_relevance_asessment

if __name__ == '__main__':
    """
    Runner for relevace asesment of users.
    """
    user_relevance_asessment(save_to_redis=False)
    save_model_variant_relevances(crop_by_date=True, last_n_by_date=66)
    save_model_variant_relevances(crop_by_date=True, last_n_by_date=581)
