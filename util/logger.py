import logging
import colorlog
import os

# Global logger instance
_logger_instance = None

class SimpleLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.logger.handlers.clear()
        
        # Extract clean module name for context
        self.context = self._get_clean_context(name)
        
        # Setup colored console handler
        console_handler = colorlog.StreamHandler()
        console_format = colorlog.ColoredFormatter(
            fmt='%(asctime)s %(log_color)s[%(levelname)s]%(reset)s %(log_color)s[%(context)s]%(reset)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            reset=True,
            log_colors={
                'DEBUG': 'blue',
                'INFO': 'green', 
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white'
            }
        )
        
        console_handler.setFormatter(console_format)
        console_handler.addFilter(self._add_context)
        self.logger.addHandler(console_handler)
    
    def _get_clean_context(self, module_name: str) -> str:
        """Extract clean context from module name"""
        if module_name == '__main__':
            return 'Main'
        
        # Get just the filename without path and extension
        if '.' in module_name:
            # Handle cases like 'mcp_server.arxiv_server'
            clean_name = module_name.split('.')[-1]
        else:
            clean_name = module_name
        
        # Convert snake_case to readable format
        # arxiv_server -> ArxivServer
        words = clean_name.split('_')
        return ''.join(word.capitalize() for word in words)
    
    def _add_context(self, record):
        """Add context to log record"""
        record.context = self.context
        return True
    
    def info(self, message: str):
        self.logger.info(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def debug(self, message: str):
        self.logger.debug(message)

def get_logger(name: str = None) -> SimpleLogger:
    """Get logger instance"""
    global _logger_instance
    if _logger_instance is None or (name and name != getattr(_logger_instance, 'original_name', None)):
        _logger_instance = SimpleLogger(name or '__main__')
        _logger_instance.original_name = name
    return _logger_instance