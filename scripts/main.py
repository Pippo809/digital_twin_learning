from digital_twin_learning.data import CostConfig, HyperParam, ModelConfig
from digital_twin_learning.utils import TaskResolver, get_args


if __name__ == "__main__":

    args = get_args()

    args = TaskResolver(args).to_args(args)
    hyper_params = HyperParam(metric=args.metric, **args.rand_pcds)
    cost_config = CostConfig(cost_path=args.cost_path, metric=args.metric, sample=args.sample, evaluate=args.evaluate_only)

    train_data, test_data = cost_config.generate_dataset(args, hyper_params)

    model = ModelConfig(args)
    train_utils_config, TrainUtils = model._get_dict(args, train_data, test_data), model._getTrainUtilsClass()

    train_utils = TrainUtils(**train_utils_config, test_data=test_data)
    train_utils.run(args)
