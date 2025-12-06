"""Custom exceptions for the AI agent."""


class InvalidModelException(Exception):
    """Exception raised when an invalid model name is provided."""

    def __init__(self, model_name: str, allowed_models: list):
        """
        Initialize the exception.

        Args:
            model_name: The invalid model name that was provided
            allowed_models: List of allowed model names
        """
        self.model_name = model_name
        self.allowed_models = allowed_models
        self.message = f"Invalid model name '{model_name}'. Allowed models: {', '.join(allowed_models)}"
        super().__init__(self.message)
