"""
Comprehensive Error Handling for Evaluation API
================================================

Handles common errors gracefully with proper logging and user-friendly messages.
"""

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from typing import Union
import logging
import traceback
from datetime import datetime
from typing import Optional

# Setup logging
logger = logging.getLogger("evaluation_api")
logger.setLevel(logging.INFO)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class EvaluationAPIException(Exception):
    """Base exception for evaluation API"""
    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ProjectNotFoundException(EvaluationAPIException):
    """Project not found"""
    def __init__(self, project_id: int):
        super().__init__(
            message=f"Project {project_id} not found",
            status_code=404,
            details={"project_id": project_id}
        )


class InvalidFilterException(EvaluationAPIException):
    """Invalid filter parameters"""
    def __init__(self, message: str, filters: dict):
        super().__init__(
            message=f"Invalid filters: {message}",
            status_code=400,
            details={"filters": filters}
        )


class CacheException(EvaluationAPIException):
    """Cache operation failed"""
    def __init__(self, operation: str, error: str):
        super().__init__(
            message=f"Cache {operation} failed: {error}",
            status_code=500,
            details={"operation": operation, "error": error}
        )


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

async def evaluation_exception_handler(request: Request, exc: EvaluationAPIException):
    """Handle custom evaluation exceptions"""
    
    logger.error(
        f"EvaluationAPIException: {exc.message}",
        extra={
            "status_code": exc.status_code,
            "details": exc.details,
            "path": request.url.path,
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "details": exc.details,
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path,
        }
    )


async def validation_exception_handler(request: Request, exc: Union[RequestValidationError, ValidationError]):
    """Handle Pydantic validation errors"""
    
    # Extract error details
    errors = []
    for error in exc.errors():
        errors.append({
            "field": " -> ".join(str(x) for x in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        })
    
    logger.warning(
        f"Validation error on {request.url.path}",
        extra={"errors": errors}
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation failed",
            "details": errors,
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path,
        }
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler"""
    
    # Log full traceback
    logger.error(
        f"Unhandled exception on {request.url.path}: {str(exc)}",
        exc_info=True,
        extra={
            "path": request.url.path,
            "method": request.method,
            "traceback": traceback.format_exc(),
        }
    )
    
    # Don't expose internal errors to users in production
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path,
        }
    )


# ============================================================================
# SAFE QUERY WRAPPER
# ============================================================================

from typing import TypeVar, Callable, Any
from functools import wraps

T = TypeVar('T')


def safe_query(default_value: T = None, log_errors: bool = True):
    """
    Decorator to safely execute database queries with error handling.
    
    Usage:
        @safe_query(default_value=[], log_errors=True)
        async def get_images(project_id: int):
            return await query_images(project_id)
    """
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(
                        f"Query failed in {func.__name__}: {str(e)}",
                        exc_info=True,
                        extra={
                            "function": func.__name__,
                            "args": args,
                            "kwargs": kwargs,
                        }
                    )
                
                # Return default value instead of failing
                return default_value
        
        return wrapper
    return decorator


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_project_access(project_id: int, user_id: Optional[int] = None) -> bool:
    """
    Validate user has access to project.
    Replace with actual permission check.
    """
    from projects.models import Project
    from django.core.exceptions import ObjectDoesNotExist
    
    try:
        project = Project.objects.get(id=project_id)
        
        # Add permission check here
        # if user_id and not project.has_access(user_id):
        #     return False
        
        return True
    except ObjectDoesNotExist:
        raise ProjectNotFoundException(project_id)


def validate_pagination(page: int, page_size: int):
    """Validate pagination parameters"""
    
    if page < 1:
        raise InvalidFilterException(
            "Page must be >= 1",
            {"page": page}
        )
    
    if page_size < 1 or page_size > 1000:
        raise InvalidFilterException(
            "Page size must be between 1 and 1000",
            {"page_size": page_size}
        )


# ============================================================================
# REGISTER HANDLERS WITH FASTAPI
# ============================================================================

def setup_exception_handlers(app):
    """
    Register all exception handlers with FastAPI app.
    
    Usage in main.py:
        from api.error_handling import setup_exception_handlers
        
        app = FastAPI()
        setup_exception_handlers(app)
    """
    
    from fastapi.exceptions import RequestValidationError
    from pydantic import ValidationError
    
    # Custom exceptions
    app.add_exception_handler(EvaluationAPIException, evaluation_exception_handler)
    
    # Validation errors
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)
    
    # Generic catch-all
    app.add_exception_handler(Exception, generic_exception_handler)
    
    logger.info("✓ Exception handlers registered")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
# In your endpoint:

from api.error_handling import validate_project_access, safe_query, ProjectNotFoundException

@router.get("/projects/{project_id}/summary")
async def get_summary(project_id: int):
    # Validate access
    validate_project_access(project_id)
    
    # Safe query with fallback
    @safe_query(default_value=DatasetEvaluationSummary())
    async def fetch_summary():
        return await query_builder.get_quick_summary(project_id)
    
    summary = await fetch_summary()
    return summary


# In main.py:

from api.error_handling import setup_exception_handlers

app = FastAPI()
setup_exception_handlers(app)
"""