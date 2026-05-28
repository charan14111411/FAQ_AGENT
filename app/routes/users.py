import re
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import get_db, find_user_by_email, create_user, get_last_session_category, write_log
from app.models import UserRequest, UserResponse
from app.logger import get_logger

logger = get_logger()
router = APIRouter()

EMAIL_REGEX = r"^[^@]+@[^@]+\.[^@]+$"

def validate_email(email: str):
    if not re.match(EMAIL_REGEX, email.strip()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format"
        )

@router.post("/users", response_model=UserResponse)
async def handle_user(req: UserRequest, db: AsyncSession = Depends(get_db)):
    try:
        validate_email(req.email)
        user = await find_user_by_email(db, req.email.strip())
        
        if user:
            user_id_str = str(user.id)
            last_category = await get_last_session_category(db, user.id)
            
            await write_log(
                db, 
                level="INFO", 
                event="user_returning", 
                message=f"Returning user: {req.email}", 
                user_id=user.id, 
                meta={"email": req.email, "last_category": last_category}
            )
            logger.info(
                f"User returning: {req.email}", 
                extra={"event": "user_returning", "user_id": user_id_str, "meta": {"last_category": last_category}}
            )
            
            return UserResponse(
                id=user_id_str,
                name=user.name,
                email=user.email,
                is_returning=True,
                last_category=last_category
            )
        else:
            new_user = await create_user(db, req.name.strip(), req.phone.strip(), req.email.strip())
            new_user_id_str = str(new_user.id)
            
            await write_log(
                db, 
                level="INFO", 
                event="user_created", 
                message=f"New user created: {req.email}", 
                user_id=new_user.id, 
                meta={"email": req.email}
            )
            logger.info(
                f"New user created: {req.email}", 
                extra={"event": "user_created", "user_id": new_user_id_str}
            )
            
            return UserResponse(
                id=new_user_id_str,
                name=new_user.name,
                email=new_user.email,
                is_returning=False,
                last_category=None
            )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in user route: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request."
        )
