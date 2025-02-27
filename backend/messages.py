import json
from enum import Enum
from dataclasses import dataclass
from typing import Union


# Define an enumeration for the message types
class MessageType(Enum):
    CHAT = "chat"
    ADD_REACTION = "add_reaction"
    REMOVE_REACTION = "remove_reaction"

# Define the structure for each message type using dataclasses
@dataclass
class ChatMessage:
    text: str
    timestamp: str
    username: str


@dataclass
class AddReactionMessage:
    id: int
    reaction: str

@dataclass
class RemoveReactionMessage:
    id: int
    reaction: str

# Define the Message class with dynamic structure based on type
class Message:
    def __init__(self, message_type: MessageType, content: Union[ChatMessage, AddReactionMessage, RemoveReactionMessage], username: str, token: str):
        self.message_type = message_type
        self.content = content
        self.username = username
        self.token = token

    @staticmethod
    def from_json(json_str: str):
        # Parse the JSON string
        data = json.loads(json_str)
        
        # Extract the username and token for authentication
        username = data['username']
        token = data['token']

        message_type = MessageType(data['type'])

        # Create a message object based on the type
        if message_type == MessageType.CHAT:
            content = ChatMessage(text=data['text'], timestamp=data['timestamp'], username=data['username'])
        elif message_type == MessageType.ADD_REACTION:
            content = AddReactionMessage(id=data['id'], reaction=data['reaction'])
        elif message_type == MessageType.REMOVE_REACTION:
            content = RemoveReactionMessage(id=data['id'], reaction=data['reaction'])
        else:
            raise ValueError(f"Unsupported message type: {message_type}")

        # Return the Message object
        return Message(message_type, content, username, token)

    def __str__(self):
        return f"Message(type={self.message_type.value}, username={self.username}, token={self.token}, content={self.content})"
    
def parse_reaction(reaction: str) -> int:
    if reaction == 'heart':
        return 1
    elif reaction == 'fire':
        return 2
    elif reaction == 'angy':
        return 3
    elif reaction == 'clown':
        return 4
    elif reaction == 'nerd':
        return 5
    elif reaction == 'rolling eyes':
        return 6
    elif reaction == 'this tbh':
        return 7
    elif reaction == 'cross':
        return 8
    
def decode_reaction(reaction: int) -> str:
    if reaction == 1:
        return 'heart'
    elif reaction == 2:
        return 'fire'
    elif reaction == 3:
        return 'angy'
    elif reaction == 4:
        return 'clown'
    elif reaction == 5:
        return 'nerd'
    elif reaction == 6:
        return 'rolling eyes'
    elif reaction == 7:
        return 'this tbh'
    elif reaction == 8:
        return 'cross'