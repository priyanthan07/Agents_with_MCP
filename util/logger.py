import logging

def get_logger(name: str) -> logging.Logger:
    """
    Create a simple, consistent logger for any module
    
    Args:
        name: Usually __name__ from the calling module
        
    Returns:
        Configured logger ready to use
    """
    # Create logger with the given name (usually the module name)
    logger = logging.getLogger(name)
    
    # Only configure if this logger hasn't been set up yet
    # This prevents duplicate handlers when importing multiple times
    if not logger.handlers:
        # Create a handler that prints to console
        handler = logging.StreamHandler()
        
        # Set a clear, readable format for log messages
        # Shows: timestamp - module_name - level - message
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add the handler to our logger
        logger.addHandler(handler)
        
        # Set logging level to INFO (shows INFO, WARNING, ERROR)
        # Change to DEBUG if you want more detailed output
        logger.setLevel(logging.INFO)
    
    return logger