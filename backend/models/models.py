from pydantic import BaseModel


class User(BaseModel):
    username: str
    password: str


class Message(BaseModel):
    text: str
    timestamp: int
    uid: str
