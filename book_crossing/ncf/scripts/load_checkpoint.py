import click

from book_crossing.ncf.models import NeuMF


@click.command()
@click.option('--path', type=click.Path(exists=True, dir_okay=False))
def check_model_loads(path: str):
    model = NeuMF.load_from_checkpoint(path)
    model.summarize()
