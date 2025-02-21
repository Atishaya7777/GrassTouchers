from models.models import Message, User
from utils.encrypt import hash_password, verify_password
from utils.jwt import create_access_token, verify_token
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from db.supabase import create_supabase_client
import json
from postgrest.exceptions import APIError

db_client = create_supabase_client()
OLD_MESSAGE_LOAD_AMOUNT = 50


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


app = FastAPI()

manager = ConnectionManager()

origins = [
    # "http://localhost",
    # "http://localhost:3000",  # This should be your frontend URL
    '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


log = []

""" AUTHENTICATION STUFF STARTS HERE """


@app.post("/login")
async def login(user: User):
    try:
        response = (
            db_client.table("users")
            .select("password_hash")
            .eq("username", user.username)
            .single()
            .execute()
        )

        if not response or not verify_password(
            user.password, response.data.get("password_hash")
        ):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token = create_access_token(user.username)

        return {
            "status": 200,
            "message": "Success",
            "data": {"token": token, "username": user.username},
        }

    except:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {"message": "Success"}


@app.post("/register")
async def register(user: User):
    password_hash = hash_password(user.password)

    try:
        db_client.table("users").insert(
            {"username": user.username, "password_hash": password_hash}
        ).execute()

        return {"message": "User created successfully"}

    except:
        raise HTTPException(status_code=400, detail="Username already exists")


""" AUTHENTICATION STUFF ENDS HERE """

@app.get("/leaderboard")
async def leaderboard(query: str = "messages"):
    if query == "messages":
        try:
            response = (db_client.table("messages").select("username").execute()
            )
            message_counts = {}
            for record in response.data:
                username = record['username']
                if username in message_counts:
                    message_counts[username] += 1
                else:
                    message_counts[username] = 1
            print("Messages sent by each user:")
            for username, count in message_counts.items():
                print(f"User ID: {username}, Messages sent: {count}")
            print(response)
        except APIError as e:
            print("Rats",e)
    elif query == "credits":
        try:
            response = (db_client.table("users")
                        .select("username,anti_social_credit")
                        .execute()
            )
            print(response)
        except APIError as e:
            print("Rats",e)


async def send_old_messages(websocket: WebSocket):
    response = None
    try:
        response = (
            db_client.table("messages")
            .select("*")
            .limit(OLD_MESSAGE_LOAD_AMOUNT)
            .execute()
        )
    except APIError as error:
        print("Failed to load messages, tell the client probably", error)
    if response != None:
        for message in response.data:
            await manager.send_personal_message(json.dumps(message), websocket)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    await send_old_messages(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                authenticated_message = json.loads(data)
                message = authenticated_message["message"]
                if "token" in authenticated_message.keys():
                    username = verify_token(authenticated_message["token"][7:])
                    if username != message["username"]:
                        raise HTTPException(
                            status_code=401,
                            detail="Invalid token, please reauthenticate!",
                        )
                    else:
                        response = db_client.table(
                            "messages").insert(message).execute()
                        await manager.broadcast(json.dumps(message))
                else:
                    raise HTTPException(
                        status_code=401, detail="Invalid token, please reauthenticate!"
                    )
            except APIError as error:
                print("Failed to send message, tell the client probably", error)

    except WebSocketDisconnect:
        manager.disconnect(websocket)


"""ANTI-SOCIAL CREDIT CODE RESIDES HEREIN"""

@app.get("/antiSocialCredit/{username}")
async def get_anti_social_credit(username: str):
    try:
        response = (
            db_client.table("users")
            .select("anti_social_credit")
            .eq("username", username)
            .execute()
        )

        print("Here's the response.data: ", response.data)

        if not response.data:
            raise HTTPException(status_code=404, detail="Invalid username")
        
        return {"username": username, "anti_social_credit": response.data[0]["anti_social_credit"]}

    except Exception as e:
        if e.status_code == 404:
            raise HTTPException(status_code=404, detail="Invalid username")
        else:
            print(f"Something went wrong: {e}")
            raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    uvicorn.run("example:app", host="127.0.0.1", port=8000, reload=True)
