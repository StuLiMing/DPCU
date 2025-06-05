import os
from omegaconf import OmegaConf

class WandbLogger:
    """
    Log using `Weights and Biases`.
    """
    def __init__(self, opt):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )
        
        self._wandb = wandb
        
        opt=OmegaConf.to_container(opt, resolve=True)
        self.opt=opt
        
        # Initialize a W&B run
        if self._wandb.run is None:
            self._wandb.init(
                project="Resshift_PCU",
                config=opt,
                dir='/amax/lm/ResShift_pcu/experiments',
                name=opt["wandb"]["name"]
            )

            
    def log_metrics(self, metrics, commit=True): 
        """
        Log train/validation metrics onto W&B.

        metrics: dictionary of metrics to be logged
        """
        self._wandb.log(metrics, commit=commit)


