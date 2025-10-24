"""
Encryption utilities for storing credentials securely.
"""
from cryptography.fernet import Fernet
from django.conf import settings
import base64
import json


def get_cipher():
    """Get Fernet cipher instance."""
    encryption_key = getattr(settings, 'ENCRYPTION_KEY', None)
    
    if not encryption_key:
        raise ValueError(
            "ENCRYPTION_KEY not set in settings. "
            "Generate one with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
        )
    
    if isinstance(encryption_key, str):
        encryption_key = encryption_key.encode()
    
    return Fernet(encryption_key)


def encrypt_credentials(credentials: dict) -> bytes:
    """
    Encrypt credentials dictionary.
    
    Args:
        credentials: Dictionary of credentials to encrypt
    
    Returns:
        Encrypted bytes
    """
    cipher = get_cipher()
    json_str = json.dumps(credentials)
    return cipher.encrypt(json_str.encode())


def decrypt_credentials(encrypted_data: bytes) -> dict:
    """
    Decrypt credentials.
    
    Args:
        encrypted_data: Encrypted credentials bytes
    
    Returns:
        Decrypted credentials dictionary
    """
    cipher = get_cipher()
    decrypted = cipher.decrypt(encrypted_data)
    return json.loads(decrypted.decode())