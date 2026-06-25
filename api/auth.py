"""
API key authentication middleware for NordSpot.

Implemented in Epic 5.
Industry pattern: each B2B client gets one API key; keys are stored
hashed in the database and verified on every request.
"""
# from fastapi import Security, HTTPException, status
# from fastapi.security import APIKeyHeader

# API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)

# async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
#     # TODO: look up hashed key in database
#     raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
