"""
Service layer for integrating perceptra-storage with Django models.
"""
from typing import Optional, Dict, Any
from django.conf import settings
import logging
import json
import os

from perceptra_storage import (
    get_storage_adapter,
    BaseStorageAdapter,
    StorageError,
    StorageConnectionError,
)
from .models import StorageProfile, SecretRef, EncryptedSecret

logger = logging.getLogger(__name__)


# Export the error class so it can be imported from services
__all__ = [
    'StorageServiceError',
    'SecretRetrievalError',
    'get_storage_adapter_for_profile',
    'test_storage_profile_connection',
    'get_default_storage_adapter',
    'validate_storage_config',
    'StorageManager',
    'retrieve_credentials_from_secret_ref',
]


class StorageServiceError(Exception):
    """Raised when storage service operations fail."""
    pass


class SecretRetrievalError(StorageServiceError):
    """Raised when secret retrieval fails."""
    pass


def retrieve_credentials_from_secret_ref(secret_ref: SecretRef) -> Dict[str, Any]:
    """
    Retrieve credentials from a SecretRef.
    
    This function implements the logic to fetch actual credentials from
    various secret management systems based on the provider.
    
    Args:
        secret_ref: SecretRef instance containing secret location info
    
    Returns:
        Dictionary containing credentials
    
    Raises:
        SecretRetrievalError: If credential retrieval fails
    """
    try:
        provider = secret_ref.provider
        
        if provider == 'vault':
            return _retrieve_from_vault(secret_ref)
        elif provider == 'azure_kv':
            return _retrieve_from_azure_kv(secret_ref)
        elif provider == 'aws_sm':
            return _retrieve_from_aws_sm(secret_ref)
        elif provider == 'local_enc':
            return _retrieve_from_local_encrypted(secret_ref)
        else:
            raise SecretRetrievalError(f"Unsupported secret provider: {provider}")
            
    except Exception as e:
        logger.error(f"Failed to retrieve credentials from {secret_ref}: {e}")
        raise SecretRetrievalError(f"Credential retrieval failed: {e}") from e


def _retrieve_from_vault(secret_ref: SecretRef) -> Dict[str, Any]:
    """
    Retrieve credentials from HashiCorp Vault.
    
    Args:
        secret_ref: SecretRef instance with Vault configuration
        
    Returns:
        Dictionary containing credentials
        
    Raises:
        SecretRetrievalError: If retrieval fails
    """
    try:
        import hvac
    except ImportError:
        raise SecretRetrievalError(
            "hvac library not installed. Install with: pip install hvac"
        )
    
    try:
        # Get Vault configuration from settings or secret_ref metadata
        vault_url = getattr(settings, 'VAULT_URL', os.getenv('VAULT_URL'))
        vault_token = getattr(settings, 'VAULT_TOKEN', os.getenv('VAULT_TOKEN'))
        vault_namespace = getattr(settings, 'VAULT_NAMESPACE', os.getenv('VAULT_NAMESPACE'))
        
        # Override with secret_ref metadata if available
        if secret_ref.metadata:
            vault_url = secret_ref.metadata.get('vault_url', vault_url)
            vault_namespace = secret_ref.metadata.get('namespace', vault_namespace)
        
        if not vault_url:
            raise SecretRetrievalError("VAULT_URL not configured")
        
        # Initialize Vault client
        client = hvac.Client(
            url=vault_url,
            token=vault_token,
            namespace=vault_namespace
        )
        
        if not client.is_authenticated():
            raise SecretRetrievalError("Vault authentication failed")
        
        # Determine KV version (default to v2)
        kv_version = secret_ref.metadata.get('kv_version', 2) if secret_ref.metadata else 2
        
        # Retrieve secret based on KV version
        if kv_version == 2:
            secret_response = client.secrets.kv.v2.read_secret_version(
                path=secret_ref.path,
                mount_point=secret_ref.metadata.get('mount_point', 'secret') if secret_ref.metadata else 'secret'
            )
            secret_data = secret_response['data']['data']
        else:  # KV v1
            secret_response = client.secrets.kv.v1.read_secret(
                path=secret_ref.path,
                mount_point=secret_ref.metadata.get('mount_point', 'secret') if secret_ref.metadata else 'secret'
            )
            secret_data = secret_response['data']
        
        # Extract specific key if provided
        if secret_ref.key:
            if secret_ref.key not in secret_data:
                raise SecretRetrievalError(f"Key '{secret_ref.key}' not found in Vault secret")
            
            # Handle JSON strings
            value = secret_data[secret_ref.key]
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return {'value': value}
            return value if isinstance(value, dict) else {'value': value}
        
        return secret_data
        
    except hvac.exceptions.VaultError as e:
        logger.error(f"Vault error retrieving secret {secret_ref.path}: {e}")
        raise SecretRetrievalError(f"Vault error: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error retrieving from Vault: {e}")
        raise SecretRetrievalError(f"Failed to retrieve from Vault: {e}") from e


def _retrieve_from_azure_kv(secret_ref: SecretRef) -> Dict[str, Any]:
    """
    Retrieve credentials from Azure Key Vault.
    
    Args:
        secret_ref: SecretRef instance with Azure KV configuration
        
    Returns:
        Dictionary containing credentials
        
    Raises:
        SecretRetrievalError: If retrieval fails
    """
    try:
        from azure.keyvault.secrets import SecretClient
        from azure.identity import DefaultAzureCredential, ClientSecretCredential
        from azure.core.exceptions import AzureError
    except ImportError:
        raise SecretRetrievalError(
            "Azure SDK not installed. Install with: pip install azure-keyvault-secrets azure-identity"
        )
    
    try:
        # Get Key Vault URL from settings or metadata
        vault_url = getattr(settings, 'AZURE_KEY_VAULT_URL', os.getenv('AZURE_KEY_VAULT_URL'))
        
        if secret_ref.metadata:
            vault_url = secret_ref.metadata.get('vault_url', vault_url)
        
        if not vault_url:
            raise SecretRetrievalError("Azure Key Vault URL not configured")
        
        # Determine authentication method
        # Note: Never store client_secret in metadata - use environment variables or managed identity
        if secret_ref.metadata and 'client_id' in secret_ref.metadata:
            # Use service principal authentication with secret from environment
            client_secret = getattr(settings, 'AZURE_CLIENT_SECRET', None)
            if not client_secret:
                raise SecretRetrievalError("AZURE_CLIENT_SECRET not configured in settings")
            
            credential = ClientSecretCredential(
                tenant_id=secret_ref.metadata.get('tenant_id'),
                client_id=secret_ref.metadata.get('client_id'),
                client_secret=client_secret
            )
        else:
            # Use default credential chain (managed identity, environment variables, etc.)
            credential = DefaultAzureCredential()
        
        # Initialize Key Vault client
        client = SecretClient(vault_url=vault_url, credential=credential)
        
        # Retrieve secret
        # Azure KV uses the 'key' field as the secret name
        secret_name = (secret_ref.key or secret_ref.path).replace('/', '-').rstrip().lower()
        secret = client.get_secret(secret_name)
        
        # Try to parse as JSON, otherwise return as string value
        try:
            return json.loads(secret.value)
        except json.JSONDecodeError:
            return {'value': secret.value}
            
    except AzureError as e:
        logger.error(f"Azure Key Vault error retrieving secret {secret_ref.key}: {e}")
        raise SecretRetrievalError(f"Azure Key Vault error: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error retrieving from Azure Key Vault: {e}")
        raise SecretRetrievalError(f"Failed to retrieve from Azure Key Vault: {e}") from e


def _retrieve_from_aws_sm(secret_ref: SecretRef) -> Dict[str, Any]:
    """
    Retrieve credentials from AWS Secrets Manager.
    
    Args:
        secret_ref: SecretRef instance with AWS SM configuration
        
    Returns:
        Dictionary containing credentials
        
    Raises:
        SecretRetrievalError: If retrieval fails
    """
    try:
        import boto3
        from botocore.exceptions import ClientError, BotoCoreError
    except ImportError:
        raise SecretRetrievalError(
            "boto3 library not installed. Install with: pip install boto3"
        )
    
    try:
        # Get AWS region from settings or metadata
        region_name = getattr(settings, 'AWS_SECRETS_MANAGER_REGION', 'us-east-1')
        
        if secret_ref.metadata:
            region_name = secret_ref.metadata.get('region', region_name)
        
        # Initialize Secrets Manager client
        # Note: AWS credentials should come from environment, IAM role, or credentials file
        # NEVER store AWS credentials in metadata
        session_kwargs = {'region_name': region_name}
        
        # Use role ARN if provided for cross-account access
        if secret_ref.metadata and 'role_arn' in secret_ref.metadata:
            sts_client = boto3.client('sts')
            assumed_role = sts_client.assume_role(
                RoleArn=secret_ref.metadata['role_arn'],
                RoleSessionName='PerceptraStorageSession'
            )
            credentials = assumed_role['Credentials']
            session_kwargs['aws_access_key_id'] = credentials['AccessKeyId']
            session_kwargs['aws_secret_access_key'] = credentials['SecretAccessKey']
            session_kwargs['aws_session_token'] = credentials['SessionToken']
        
        client = boto3.client('secretsmanager', **session_kwargs)
        
        # Retrieve secret
        # The 'path' field contains the secret name/ARN
        response = client.get_secret_value(SecretId=secret_ref.path)
        
        # Parse the secret string
        if 'SecretString' in response:
            secret_string = response['SecretString']
            try:
                secret_dict = json.loads(secret_string)
                
                # If a specific key is requested, extract it
                if secret_ref.key:
                    if secret_ref.key not in secret_dict:
                        raise SecretRetrievalError(
                            f"Key '{secret_ref.key}' not found in AWS secret"
                        )
                    value = secret_dict[secret_ref.key]
                    return value if isinstance(value, dict) else {'value': value}
                
                return secret_dict
            except json.JSONDecodeError:
                # Secret is plain text, not JSON
                return {'value': secret_string}
        else:
            # Binary secret
            import base64
            binary_secret = response['SecretBinary']
            return {'value': base64.b64encode(binary_secret).decode('utf-8')}
            
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"AWS Secrets Manager error ({error_code}): {e}")
        
        if error_code == 'ResourceNotFoundException':
            raise SecretRetrievalError(f"Secret not found: {secret_ref.path}") from e
        elif error_code == 'AccessDeniedException':
            raise SecretRetrievalError(f"Access denied to secret: {secret_ref.path}") from e
        else:
            raise SecretRetrievalError(f"AWS Secrets Manager error: {e}") from e
            
    except BotoCoreError as e:
        logger.error(f"boto3 error retrieving secret: {e}")
        raise SecretRetrievalError(f"AWS SDK error: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error retrieving from AWS Secrets Manager: {e}")
        raise SecretRetrievalError(f"Failed to retrieve from AWS Secrets Manager: {e}") from e


def _retrieve_from_local_encrypted(secret_ref: SecretRef) -> Dict[str, Any]:
    """
    Retrieve credentials from local encrypted storage.
    
    IMPORTANT: This stores encrypted secrets in a separate EncryptedSecret model,
    NOT in the SecretRef metadata. The encryption key MUST be in Django settings
    or environment variables, never in the database.
    
    Args:
        secret_ref: SecretRef instance with local encrypted configuration
        
    Returns:
        Dictionary containing credentials
        
    Raises:
        SecretRetrievalError: If retrieval or decryption fails
    """
    try:
        from cryptography.fernet import Fernet, InvalidToken
    except ImportError:
        raise SecretRetrievalError(
            "cryptography library not installed. Install with: pip install cryptography"
        )
    
    try:
        # Get encryption key from settings (NEVER from database/metadata)
        encryption_key = getattr(settings, 'SECRET_ENCRYPTION_KEY', None)
        
        if not encryption_key:
            raise SecretRetrievalError(
                "SECRET_ENCRYPTION_KEY not configured in settings. "
                "Generate one with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
            )
        
        # Initialize Fernet cipher
        fernet = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
        
        # Import here to avoid circular imports
        from .models import EncryptedSecret
        
        # Retrieve the encrypted secret from the database
        # The 'path' field is used as the identifier for the encrypted secret
        try:
            encrypted_secret = EncryptedSecret.objects.get(
                organization=secret_ref.organization,
                identifier=secret_ref.path
            )
        except EncryptedSecret.DoesNotExist:
            raise SecretRetrievalError(
                f"No encrypted secret found with identifier: {secret_ref.path}"
            )
        
        # Decrypt the value
        encrypted_value = encrypted_secret.encrypted_value
        if isinstance(encrypted_value, str):
            encrypted_value = encrypted_value.encode()
        
        decrypted_bytes = fernet.decrypt(encrypted_value)
        decrypted_string = decrypted_bytes.decode('utf-8')
        
        # Try to parse as JSON
        try:
            secret_data = json.loads(decrypted_string)
            
            # If a specific key is requested, extract it
            if secret_ref.key and isinstance(secret_data, dict):
                if secret_ref.key not in secret_data:
                    raise SecretRetrievalError(
                        f"Key '{secret_ref.key}' not found in decrypted secret"
                    )
                value = secret_data[secret_ref.key]
                return value if isinstance(value, dict) else {'value': value}
            
            
            logging.info(secret_data)
            return secret_data if isinstance(secret_data, dict) else {'value': secret_data}
            
        except json.JSONDecodeError:
            # Not JSON, return as plain value
            return {'value': decrypted_string}
            
    except InvalidToken as e:
        logger.error("Failed to decrypt secret - invalid token or corrupted data")
        raise SecretRetrievalError(
            "Decryption failed - invalid encryption key or corrupted data"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error retrieving from local encrypted storage: {e}")
        raise SecretRetrievalError(
            f"Failed to retrieve from local encrypted storage: {e}"
        ) from e


# Utility functions for managing locally encrypted secrets
def create_encrypted_secret(
    organization,
    identifier: str,
    secret_data: Dict[str, Any],
    description: str = None,
    encryption_key: str = None
) -> 'EncryptedSecret':
    """
    Create and store an encrypted secret in the database.
    
    Args:
        organization: Organization instance that owns the secret
        identifier: Unique identifier for the secret (used in SecretRef.path)
        secret_data: Dictionary containing secret data to encrypt
        description: Optional description of the secret
        encryption_key: Optional encryption key (uses settings if not provided)
        
    Returns:
        EncryptedSecret instance
        
    Example:
        >>> from organizations.models import Organization
        >>> org = Organization.objects.get(name='MyOrg')
        >>> secret = create_encrypted_secret(
        ...     organization=org,
        ...     identifier='s3-prod-credentials',
        ...     secret_data={'access_key': 'AKIAXX...', 'secret_key': 'xxx...'},
        ...     description='Production S3 credentials'
        ... )
        >>> # Now create a SecretRef pointing to this
        >>> secret_ref = SecretRef.objects.create(
        ...     organization=org,
        ...     provider='local_enc',
        ...     path='s3-prod-credentials',
        ...     key='access_key'  # or leave empty to get all data
        ... )
    """
    from cryptography.fernet import Fernet
    from .models import EncryptedSecret
    
    if encryption_key is None:
        encryption_key = getattr(settings, 'SECRET_ENCRYPTION_KEY', None)
        if not encryption_key:
            raise ValueError("SECRET_ENCRYPTION_KEY not configured in settings")
    
    fernet = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
    
    # Convert to JSON string
    secret_string = json.dumps(secret_data)
    
    # Encrypt
    encrypted_bytes = fernet.encrypt(secret_string.encode())
    encrypted_value = encrypted_bytes.decode('utf-8')
    
    # Create or update the encrypted secret
    encrypted_secret, created = EncryptedSecret.objects.update_or_create(
        organization=organization,
        identifier=identifier,
        defaults={
            'encrypted_value': encrypted_value,
            'description': description or ''
        }
    )
    
    logger.info(f"{'Created' if created else 'Updated'} encrypted secret: {identifier}")
    return encrypted_secret


def update_encrypted_secret(
    organization,
    identifier: str,
    secret_data: Dict[str, Any],
    encryption_key: str = None
) -> 'EncryptedSecret':
    """
    Update an existing encrypted secret.
    
    Args:
        organization: Organization instance
        identifier: Identifier of the secret to update
        secret_data: New secret data
        encryption_key: Optional encryption key (uses settings if not provided)
        
    Returns:
        Updated EncryptedSecret instance
    """
    from .models import EncryptedSecret
    
    try:
        encrypted_secret = EncryptedSecret.objects.get(
            organization=organization,
            identifier=identifier
        )
    except EncryptedSecret.DoesNotExist:
        raise ValueError(f"No encrypted secret found with identifier: {identifier}")
    
    return create_encrypted_secret(
        organization=organization,
        identifier=identifier,
        secret_data=secret_data,
        description=encrypted_secret.description,
        encryption_key=encryption_key
    )


def delete_encrypted_secret(organization, identifier: str) -> bool:
    """
    Delete an encrypted secret and all associated SecretRefs.
    
    Args:
        organization: Organization instance
        identifier: Identifier of the secret to delete
        
    Returns:
        True if deleted, False if not found
    """
    from .models import EncryptedSecret
    
    try:
        encrypted_secret = EncryptedSecret.objects.get(
            organization=organization,
            identifier=identifier
        )
        # Delete associated SecretRefs
        SecretRef.objects.filter(
            organization=organization,
            provider='local_enc',
            path=identifier
        ).delete()
        
        encrypted_secret.delete()
        logger.info(f"Deleted encrypted secret: {identifier}")
        return True
    except EncryptedSecret.DoesNotExist:
        return False


# Utility function to encrypt and store a secret locally
def store_encrypted_secret(secret_data: Dict[str, Any], encryption_key: str = None) -> str:
    """
    Helper function to encrypt secret data for local storage.
    
    Args:
        secret_data: Dictionary containing secret data
        encryption_key: Optional encryption key (uses settings if not provided)
        
    Returns:
        Base64-encoded encrypted string to store in metadata
        
    Example:
        >>> secret = {'api_key': 'abc123', 'api_secret': 'xyz789'}
        >>> encrypted = store_encrypted_secret(secret)
        >>> # Store encrypted in SecretRef.metadata['encrypted_value']
    """
    from cryptography.fernet import Fernet
    
    if encryption_key is None:
        encryption_key = getattr(settings, 'SECRET_ENCRYPTION_KEY', None)
        if not encryption_key:
            raise ValueError("SECRET_ENCRYPTION_KEY not configured")
    
    fernet = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
    
    # Convert to JSON string
    secret_string = json.dumps(secret_data)
    
    # Encrypt
    encrypted_bytes = fernet.encrypt(secret_string.encode())
    
    # Return as base64 string for storage
    return encrypted_bytes.decode('utf-8')


def get_storage_adapter_for_profile(
    profile: StorageProfile,
    test_connection: bool = False
) -> BaseStorageAdapter:
    """
    Create a storage adapter instance from a StorageProfile.
    
    Args:
        profile: StorageProfile model instance
        test_connection: Whether to test connection before returning adapter
    
    Returns:
        Initialized storage adapter
    
    Raises:
        StorageServiceError: If adapter creation or connection test fails
    
    Example:
        >>> profile = StorageProfile.objects.get(tenant=tenant, is_default=True)
        >>> adapter = get_storage_adapter_for_profile(profile)
        >>> adapter.upload_file(file_obj, 'data.csv')
    """
    try:
        # Retrieve credentials if needed
        credentials = None
        if profile.credential_ref:
            credentials = retrieve_credentials_from_secret_ref(profile.credential_ref)
        
        # Create adapter
        adapter = get_storage_adapter(
            backend=profile.backend,
            config=profile.config,
            credentials=credentials
        )
        
        # Test connection if requested
        if test_connection:
            try:
                adapter.test_connection()
                logger.info(f"Successfully connected to storage profile: {profile.name}")
            except StorageConnectionError as e:
                logger.error(f"Connection test failed for profile {profile.name}: {e}")
                raise StorageServiceError(
                    f"Storage connection test failed: {e}"
                ) from e
        
        return adapter
        
    except StorageError as e:
        logger.error(f"Failed to create adapter for profile {profile.name}: {e}")
        raise StorageServiceError(
            f"Failed to create storage adapter: {e}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error creating adapter for profile {profile.name}: {e}")
        raise StorageServiceError(
            f"Unexpected error: {e}"
        ) from e


def test_storage_profile_connection(profile: StorageProfile) -> tuple[bool, Optional[str]]:
    """
    Test connection to a storage profile.
    
    Args:
        profile: StorageProfile to test
    
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    
    Example:
        >>> success, error = test_storage_profile_connection(profile)
        >>> if not success:
        >>>     print(f"Connection failed: {error}")
    """
    try:
        adapter = get_storage_adapter_for_profile(profile, test_connection=True)
        return True, None
        
    except StorageServiceError as e:
        return False, str(e)
    except Exception as e:
        logger.exception(f"Unexpected error testing profile {profile.name}")
        return False, f"Unexpected error: {e}"


def get_default_storage_adapter(tenant) -> Optional[BaseStorageAdapter]:
    """
    Get storage adapter for tenant's default storage profile.
    
    Args:
        tenant: Tenant instance
    
    Returns:
        Storage adapter for default profile, or None if no default exists
    
    Raises:
        StorageServiceError: If adapter creation fails
    """
    try:
        profile = StorageProfile.objects.get(tenant=tenant, is_default=True)
        return get_storage_adapter_for_profile(profile)
        
    except StorageProfile.DoesNotExist:
        logger.warning(f"No default storage profile for tenant: {tenant.name}")
        return None
    except StorageProfile.MultipleObjectsReturned:
        logger.error(f"Multiple default storage profiles for tenant: {tenant.name}")
        raise StorageServiceError(
            f"Multiple default storage profiles found for tenant {tenant.name}"
        )


def validate_storage_config(backend: str, config: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate storage configuration without creating an adapter.
    
    Args:
        backend: Backend type
        config: Configuration dictionary
    
    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str])
    
    Example:
        >>> config = {'bucket_name': 'test', 'region': 'us-east-1'}
        >>> is_valid, error = validate_storage_config('s3', config)
        >>> if not is_valid:
        >>>     print(f"Invalid config: {error}")
    """
    try:
        # Try to create adapter without credentials (will validate config)
        adapter = get_storage_adapter(backend, config, credentials=None)
        return True, None
        
    except ValueError as e:
        return False, str(e)
    except Exception as e:
        logger.exception(f"Unexpected error validating config for {backend}")
        return False, f"Validation error: {e}"


class StorageManager:
    """
    Context manager for storage operations with automatic cleanup.
    
    Example:
        >>> with StorageManager(profile) as storage:
        >>>     storage.upload_file(file_obj, 'data.csv')
        >>>     data = storage.download_file('data.csv')
    """
    
    def __init__(self, profile: StorageProfile, test_connection: bool = False):
        """
        Initialize storage manager.
        
        Args:
            profile: StorageProfile to use
            test_connection: Whether to test connection on enter
        """
        self.profile = profile
        self.test_connection_flag = test_connection
        self.adapter: Optional[BaseStorageAdapter] = None
    
    def __enter__(self) -> BaseStorageAdapter:
        """Enter context and return storage adapter."""
        self.adapter = get_storage_adapter_for_profile(
            self.profile,
            test_connection=self.test_connection_flag
        )
        return self.adapter
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and perform cleanup."""
        # Cleanup if needed (close connections, etc.)
        self.adapter = None
        
        # Don't suppress exceptions
        return False