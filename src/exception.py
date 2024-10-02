"""
exception.py

This module defines a custom exception class and a utility function to 
format error messages with detailed information about the error context.

Functions:
- error_message_detail(error, error_detail: sys): 
    Returns a formatted string containing the filename, line number, 
    and message of the provided error.

Classes:
- CustomException: 
    A custom exception that stores detailed error messages using 
    the error_message_detail function.
"""

import sys

def error_message_detail(error,error_detail:sys):
    """
    Returns a formatted error message with details about the error.

    Args:
        error: The exception object.
        error_detail: The sys module, used to extract traceback information.

    Returns:
        str: A formatted string containing the filename, line number, 
             and error message.
    """
    _,_,exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occured in python script name [{filename}]\
    line number [{exc_tb.tb_lineno}] error message [{str(error)}] "
    return error_message

class CustomException(Exception):
    """
    Custom exception class that captures detailed error messages.

    Args:
        error_message: The error message to be logged.
        error_detail: The sys module, used to extract traceback information.
    """
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail)

    def __str__(self):
        return self.error_message
