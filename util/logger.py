import logging
import json
from datetime import datetime
from typing import Optional, Any, Dict
import colorlog

# Global logger instance to prevent duplicate handlers
_logger_instance = None

class AppLoggerService:
    """
    Advanced logging service equivalent to Winston logger with colored console output
    and context tracking. Displays logs only in CLI without file storage.
    """
    
    def __init__(self):
        # Define custom colors for different log levels
        self.log_colors = {
            'DEBUG': 'blue',
            'INFO': 'green', 
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
            'VERBOSE': 'purple'
        }
        
        # Create the main logger
        self.logger = logging.getLogger('MultiAgentSystem')
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent propagation to root logger to avoid duplicates
        self.logger.propagate = False
        
        # Clear any existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Setup console handler with colors
        self._setup_console_handler()
        
        # Add custom VERBOSE level (like winston's verbose)
        self._add_custom_levels()
        
    def _add_custom_levels(self):
        """Add custom logging levels to match Winston functionality"""
        # Add VERBOSE level (between INFO and DEBUG)
        logging.addLevelName(15, 'VERBOSE')
        
        def verbose(self, message, *args, **kwargs):
            if self.isEnabledFor(15):
                self._log(15, message, args, **kwargs)
        
        logging.Logger.verbose = verbose
        
    def _setup_console_handler(self):
        """Setup colored console handler similar to Winston's console transport"""
        console_handler = colorlog.StreamHandler()
        
        # Custom format with colors, timestamp, and context info
        console_format = colorlog.ColoredFormatter(
            fmt='%(asctime)s %(log_color)s[%(context)s]%(reset)s %(log_color)s%(levelname)s%(reset)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            reset=True,
            log_colors=self.log_colors,
            secondary_log_colors={},
            style='%'
        )
        
        console_handler.setFormatter(console_format)
        console_handler.setLevel(logging.DEBUG)
        
        # Add custom filter to ensure context is always present
        console_handler.addFilter(self._context_filter)
        
        self.logger.addHandler(console_handler)
        

    def _context_filter(self, record):
        """Filter to add default context if not present"""
        if not hasattr(record, 'context'):
            record.context = record.name or 'MultiAgentSystem'
        return True

        
    def _create_log_record(self, level: int, message: str, context: Optional[str] = None, 
                          trace: Optional[str] = None,
                          extra_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a log record with all the metadata"""
        extra = {}
        
        if context:
            extra['context'] = context
        if trace:
            extra['trace'] = trace
        if extra_data:
            extra['extra_data'] = extra_data
            
        return extra
        
    def log(self, message: str, context: Optional[str] = None) -> None:
        """Log an info level message"""
        extra = self._create_log_record(logging.INFO, message, context)
        self.logger.info(message, extra=extra)
        
    def info(self, message: str, context: Optional[str] = None) -> None:
        """Log an info level message"""
        extra = self._create_log_record(logging.INFO, message, context)
        self.logger.info(message, extra=extra)
        
    def error(self, message: str, trace: Optional[str] = None, context: Optional[str] = None) -> None:
        """Log an error level message with optional stack trace"""
        extra = self._create_log_record(logging.ERROR, message, context, trace)
        self.logger.error(message, extra=extra)
        
    def warn(self, message: str, context: Optional[str] = None) -> None:
        """Log a warning level message"""
        extra = self._create_log_record(logging.WARNING, message, context)
        self.logger.warning(message, extra=extra)
        
    def warning(self, message: str, context: Optional[str] = None) -> None:
        """Alias for warn method"""
        self.warn(message, context)
        
    def debug(self, message: str, context: Optional[str] = None) -> None:
        """Log a debug level message"""
        extra = self._create_log_record(logging.DEBUG, message, context)
        self.logger.debug(message, extra=extra)
        
    def verbose(self, message: str, context: Optional[str] = None) -> None:
        """Log a verbose level message (custom level between INFO and DEBUG)"""
        extra = self._create_log_record(15, message, context)  # 15 is VERBOSE level
        self.logger.log(15, message, extra=extra)
        
    def critical(self, message: str, context: Optional[str] = None) -> None:
        """Log a critical level message"""
        extra = self._create_log_record(logging.CRITICAL, message, context)
        self.logger.critical(message, extra=extra)
        
    def set_level(self, level: str) -> None:
        """Set the logging level dynamically"""
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL,
            'VERBOSE': 15
        }
        
        if level.upper() in level_map:
            self.logger.setLevel(level_map[level.upper()])
        else:
            self.logger.warning(f"Unknown log level: {level}")
            
    def get_logger(self, name: str = None):
        """Get a child logger with optional name"""
        if name:
            return self.logger.getChild(name)
        return self.logger


def get_logger(name: str = None) -> AppLoggerService:
    """
    Create a singleton logger instance to prevent duplicate handlers.
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = AppLoggerService()
    return _logger_instance