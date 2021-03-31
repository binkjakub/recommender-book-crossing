import click
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger

from book_crossing.ncf.data import BookCrossingDM
from book_crossing.ncf.models import MLP, GMF


@click.command()
@click.option('--dataset-path', type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path of interaction matrix CSV file")
@click.option('--log-dir', type=click.Path(file_okay=False), required=True,
              help="Directory where logs will be stored")
@click.option('--experiment-name', type=click.STRING, default='test',
              help="Name of experiment passed to the logger")
def run_ncf_training(dataset_path: str, log_dir: str, experiment_name: str):

    config = {
        'users_fraction': 1,
        # 'users_fraction': 0.05,
        # 'num_users': 372,
        # 'num_items': 10292,
        'num_negatives': 4,
        'batch_size': 128,
        'book_interactions_cutoff': 10,
        'user_interaction_cutoff': 10,
        'num_workers': 8,
        'learning_rate': 1e-3,
        'latent_dim_mf': 8,
        'latent_dim_mlp': 8,
        'num_users': 7441,
        'num_items': 18309,
        'layers': [16, 32, 16, 8],
        'max_epochs': 32,
    }

    seed_everything(1234)
    book_crossing_dm = BookCrossingDM(dataset_path, config)

    # model = NeuMF(config)
    # model = GMF(config)
    model = MLP(config)

    experiment_name = f'{type(model).__name__}__{experiment_name}'
    trainer = Trainer(
        max_epochs=config['max_epochs'],
        logger=_get_loggers(log_dir, experiment_name),
        default_root_dir=log_dir,
        deterministic=True,
    )
    trainer.fit(model, datamodule=book_crossing_dm)


def _get_loggers(log_dir: str, experiment_name: str) -> LightningLoggerBase:
    return WandbLogger(experiment_name, log_dir, project='recsys_ncf', log_model=True)


run_ncf_training()
