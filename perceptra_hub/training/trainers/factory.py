"""
Trainer factory and registry.
Clean way to instantiate trainers.

Location: training/trainers/factory.py
"""
from typing import Dict, Type, Optional
import logging

from .base import BaseTrainer, TrainingConfig, TrainingCallbacks

logger = logging.getLogger(__name__)

# Maps human-readable size labels to per-framework variant strings
_MODEL_SIZE_MAPPINGS: Dict[str, Dict[str, str]] = {
    'ultralytics': {
        'nano': 'n', 'small': 's', 'medium': 'm', 'large': 'l', 'xlarge': 'x',
        # pass-through single-char variants
        'n': 'n', 's': 's', 'm': 'm', 'l': 'l', 'x': 'x',
    },
    'yolo': {
        'nano': 'n', 'small': 's', 'medium': 'm', 'large': 'l', 'xlarge': 'x',
        'n': 'n', 's': 's', 'm': 'm', 'l': 'l', 'x': 'x',
    },
    'rfdetr': {
        'small': 'rfdetr_base', 'medium': 'rfdetr_base', 'large': 'rfdetr_large',
        'rfdetr_base': 'rfdetr_base', 'rfdetr_large': 'rfdetr_large',
    },
    'rf-detr': {
        'small': 'rfdetr_base', 'medium': 'rfdetr_base', 'large': 'rfdetr_large',
        'rfdetr_base': 'rfdetr_base', 'rfdetr_large': 'rfdetr_large',
    },
    'rtdetr': {
        'small': 'rfdetr_base', 'medium': 'rfdetr_base', 'large': 'rfdetr_large',
        'rfdetr_base': 'rfdetr_base', 'rfdetr_large': 'rfdetr_large',
    },
    'rt-detr': {
        'small': 'rfdetr_base', 'medium': 'rfdetr_base', 'large': 'rfdetr_large',
        'rfdetr_base': 'rfdetr_base', 'rfdetr_large': 'rfdetr_large',
    },
}


def _normalize_model_size(framework: str, size: str) -> str:
    """Translate human-readable size to the variant string expected by the trainer."""
    mapping = _MODEL_SIZE_MAPPINGS.get(framework.lower(), {})
    return mapping.get(size.lower(), size) if size else size


def validate_trainer_config(config: TrainingConfig) -> None:
    """Raise ValueError if essential training config fields are missing or invalid."""
    if config.epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {config.epochs}")
    if config.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {config.batch_size}")
    if config.learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {config.learning_rate}")
    if not config.dataset_path:
        raise ValueError("dataset_path is required")
    if not config.output_dir:
        raise ValueError("output_dir is required")


class TrainerRegistry:
    """
    Registry for trainer classes.
    Makes it easy to add new trainers without modifying core code.
    """
    
    _trainers: Dict[str, Type[BaseTrainer]] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a trainer.
        
        Usage:
            @TrainerRegistry.register('yolo')
            class YOLOTrainer(BaseTrainer):
                ...
        """
        def wrapper(trainer_class: Type[BaseTrainer]):
            if not issubclass(trainer_class, BaseTrainer):
                raise TypeError(
                    f"{trainer_class.__name__} must inherit from BaseTrainer"
                )
            
            cls._trainers[name.lower()] = trainer_class
            logger.info(f"Registered trainer: {name} -> {trainer_class.__name__}")
            return trainer_class
        
        return wrapper
    
    @classmethod
    def get(cls, name: str) -> Type[BaseTrainer]:
        """Get trainer class by name"""
        name = name.lower()
        if name not in cls._trainers:
            raise ValueError(
                f"Unknown trainer: {name}. "
                f"Available trainers: {list(cls._trainers.keys())}"
            )
        return cls._trainers[name]
    
    @classmethod
    def list_trainers(cls) -> list[str]:
        """List all registered trainers"""
        return list(cls._trainers.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if trainer is registered"""
        return name.lower() in cls._trainers


# Auto-register trainers on import
def _register_builtin_trainers():
    """Register built-in trainers"""
    try:
        from .yolo_trainer import YOLOTrainer
        TrainerRegistry.register('ultralytics')(YOLOTrainer)
        TrainerRegistry.register('yolo')(YOLOTrainer)
        TrainerRegistry.register('yolov8')(YOLOTrainer)
        TrainerRegistry.register('yolov9')(YOLOTrainer)
        TrainerRegistry.register('yolov10')(YOLOTrainer)
        TrainerRegistry.register('yolo11')(YOLOTrainer)
    except ImportError as e:
        logger.warning(f"Could not register YOLO trainer: {e}")
    
    try:
        from .rfdetr_trainer import RFDETRTrainer
        TrainerRegistry.register('rf-detr')(RFDETRTrainer)
        TrainerRegistry.register('rfdetr')(RFDETRTrainer)
        TrainerRegistry.register('rtdetr')(RFDETRTrainer)
        TrainerRegistry.register('rt-detr')(RFDETRTrainer)
    except ImportError as e:
        logger.warning(f"Could not register RF-DETR trainer: {e}")


# Register trainers on module import
_register_builtin_trainers()


class TrainerFactory:
    """
    Factory for creating trainer instances.
    Provides clean API for instantiation.
    """
    
    @staticmethod
    def create_trainer(
        framework: str,
        task: str,
        config: TrainingConfig,
        callbacks: Optional[TrainingCallbacks] = None
    ) -> BaseTrainer:
        """
        Create trainer instance.
        
        Args:
            framework: Framework name (yolo, rf-detr, pytorch, etc.)
            task: Task type (object-detection, classification, etc.)
            config: Training configuration
            callbacks: Optional callbacks for progress tracking
        
        Returns:
            Trainer instance ready for training
        
        Example:
            config = TrainingConfig(
                dataset_path='/path/to/dataset',
                output_dir='/path/to/output',
                epochs=100,
                batch_size=16
            )
            
            trainer = TrainerFactory.create_trainer(
                framework='yolo',
                task='object-detection',
                config=config
            )
            
            result = trainer.train()
        """
        if not TrainerRegistry.is_registered(framework):
            raise ValueError(
                f"Unknown framework: {framework}. "
                f"Available: {TrainerRegistry.list_trainers()}\n"
                f"To add a new trainer:\n"
                f"1. Create trainer class inheriting BaseTrainer\n"
                f"2. Register with @TrainerRegistry.register('name')"
            )
        
        trainer_class = TrainerRegistry.get(framework)

        # Validate config before instantiation
        validate_trainer_config(config)

        # Normalize model_size in model_params to the framework-specific variant string
        if config.model_params.get('model_size'):
            config.model_params['model_size'] = _normalize_model_size(
                framework, config.model_params['model_size']
            )

        logger.info(
            f"Creating trainer: {trainer_class.__name__} for task: {task}, "
            f"model_size={config.model_params.get('model_size', 'default')}"
        )

        return trainer_class(task=task, config=config, callbacks=callbacks)
    
    @staticmethod
    def from_dict(config_dict: dict) -> BaseTrainer:
        """
        Create trainer from configuration dictionary.
        Convenient for loading from JSON/YAML.
        
        Args:
            config_dict: Configuration dictionary with keys:
                - framework: str
                - task: str
                - dataset_path: str
                - output_dir: str
                - epochs: int
                - batch_size: int
                - ... (other TrainingConfig fields)
        
        Returns:
            Trainer instance
        """
        from pathlib import Path
        
        # Extract trainer-specific fields
        framework = config_dict.pop('framework')
        task = config_dict.pop('task')
        
        # Extract model params if present
        model_params = config_dict.pop('model_params', {})
        
        # Convert paths
        if 'dataset_path' in config_dict:
            config_dict['dataset_path'] = Path(config_dict['dataset_path'])
        if 'output_dir' in config_dict:
            config_dict['output_dir'] = Path(config_dict['output_dir'])
        if 'checkpoint_path' in config_dict and config_dict['checkpoint_path']:
            config_dict['checkpoint_path'] = Path(config_dict['checkpoint_path'])
        
        # Create config
        config_dict['model_params'] = model_params
        config = TrainingConfig(**config_dict)
        
        return TrainerFactory.create_trainer(framework, task, config)


# Convenience function
def get_trainer(
    framework: str,
    task: str,
    dataset_path: str,
    output_dir: str,
    **kwargs
) -> BaseTrainer:
    """
    Convenience function to create trainer with minimal arguments.
    
    Example:
        trainer = get_trainer(
            framework='yolo',
            task='object-detection',
            dataset_path='/data/coco',
            output_dir='/output',
            epochs=100,
            batch_size=16,
            model_size='n'  # YOLO-specific param
        )
    """
    from pathlib import Path
    
    # Separate model params from training config
    model_params = {}
    config_kwargs = {}
    
    # Known TrainingConfig fields
    training_config_fields = {
        'epochs', 'batch_size', 'learning_rate', 'image_size',
        'device', 'workers', 'optimizer', 'weight_decay', 'momentum',
        'scheduler', 'patience', 'augmentation', 'checkpoint_path'
    }
    
    for key, value in kwargs.items():
        if key in training_config_fields:
            config_kwargs[key] = value
        else:
            model_params[key] = value
    
    config = TrainingConfig(
        dataset_path=Path(dataset_path),
        output_dir=Path(output_dir),
        model_params=model_params,
        **config_kwargs
    )
    
    return TrainerFactory.create_trainer(framework, task, config)