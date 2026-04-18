import wandb

def init_logger(project_name, config):
    wandb.init(
        project=project_name,
        config=config,
        group="baseline_dit", 
    )

def log_metrics(metrics, step):
    wandb.log(metrics, step=step)

def log_images(images, masks, predictions, step):
    wandb.log({
        "samples": [wandb.Image(img, caption=f"Step {step}") for img in predictions]
    }, step=step)