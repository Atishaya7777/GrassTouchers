from models.models import Message, User
from utils.encrypt import hash_password, verify_password
from utils.jwt import create_access_token, verify_token
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from db.supabase import create_supabase_client
import json
from postgrest.exceptions import APIError
from messages import Message, MessageType, parse_reaction, decode_reaction
from ai.run_model import classify

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
            print(connection)
            await connection.send_text(message)


from ai import test

app = FastAPI()

manager = ConnectionManager()

origins = [
    # "http://localhost",
    # "http://localhost:3000",  # This should be your frontend URL
    "*"
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


@app.get("/aitesting/")
async def aitesting():
    return {"message": test.test()}

log = []

""" AUTHENTICATION STUFF STARTS HERE """


@app.post("/logout")
async def logout():
    return {"message": "Success"}


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
            response = db_client.table("messages").select("username").execute()
            responseUsers = db_client.table(
                "users").select("username").execute()
            message_counts = {}
            result = {}
            for record in response.data:
                username = record["username"]
                if username in message_counts:
                    message_counts[username] += 1
                else:
                    message_counts[username] = 1

            for username, count in message_counts.items():
                result[username] = count

            for record in responseUsers.data:
                username = record["username"]
                if username not in result:
                    result[username] = 0

            return {"payload": result}
        except APIError as e:
            print("Rats", e)
    elif query == "credits":
        try:
            response = (
                db_client.table("users").select(
                    "username,anti_social_credit").execute()
            )

            result = {}

            for record in response.data:
                result[record["username"]] = record["anti_social_credit"]

            return {"payload": result}
        except APIError as e:
            print("Rats", e)


async def send_old_messages(websocket: WebSocket):
    messages = None
    reactions = None
    try:
        messages = (
            db_client.table("messages")
            .select("*")
            .limit(OLD_MESSAGE_LOAD_AMOUNT)
            .execute()
        )
        reactions = (
            db_client.table("reactions")
            .select("*")
            .limit(OLD_MESSAGE_LOAD_AMOUNT)
            .execute()
        )
        for reaction in reactions.data:
            for message in messages.data:
                if reaction["message_id"] == message["message_id"]:
                    print("Found", reaction["reaction"],
                          "for", message["message_id"])
                    decoded_reaction = decode_reaction(reaction["reaction"])
                    if "reactions" not in message.keys():
                        message["reactions"] = {}
                    if decoded_reaction not in message["reactions"].keys():
                        message["reactions"][decoded_reaction] = []
                    message["reactions"][decoded_reaction].append(
                        reaction["username"])

        # print("Messages with reactions:", messages)

    except APIError as error:
        print("Failed to load messages, tell the client probably", error)
    if messages != None:
        for message in messages.data:
            await manager.send_personal_message(json.dumps(message), websocket)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    await send_old_messages(websocket)
    try:
        while True:
            data = await websocket.receive_text()

            message = Message.from_json(data)
            authenticated = verify_token(message.token) == message.username

            if authenticated:
                if message.message_type == MessageType.CHAT:
                    categories = classify(message.content.text)
                    dad_joke = categories[0]
                    nerdy = categories[1]
                    positive = categories[2]
                    negative = categories[3]
                    neutral = categories[4]
                    brainrot = categories[5]

                    if (
                        dad_joke > nerdy
                        and dad_joke > positive
                        and dad_joke > negative
                        and dad_joke > neutral
                        and dad_joke > brainrot
                    ):
                        most_likely_category = "dad_joke"
                    elif (
                        nerdy > dad_joke
                        and nerdy > positive
                        and nerdy > negative
                        and nerdy > neutral
                        and nerdy > brainrot
                    ):
                        most_likely_category = "nerdy"
                    elif (
                        positive > dad_joke
                        and positive > nerdy
                        and positive > negative
                        and positive > neutral
                        and positive > brainrot
                    ):
                        most_likely_category = "positive"
                    elif (
                        negative > dad_joke
                        and negative > nerdy
                        and negative > positive
                        and negative > neutral
                        and negative > brainrot
                    ):
                        most_likely_category = "negative"
                    elif (
                        neutral > dad_joke
                        and neutral > nerdy
                        and neutral > positive
                        and neutral > negative
                        and neutral > brainrot
                    ):
                        most_likely_category = "neutral"
                    elif (
                        brainrot > dad_joke
                        and brainrot > nerdy
                        and brainrot > positive
                        and brainrot > negative
                        and brainrot > neutral
                    ):
                        most_likely_category = "brainrot"

                    db_message = {
                        "username": message.username,
                        "text": message.content.text,
                        "timestamp": message.content.timestamp,
                        "classification": most_likely_category,
                    }

                    response = db_client.table(
                        "messages").insert(db_message).execute()
                    db_message["message_id"] = response.data[0]["message_id"]

                    await manager.broadcast(json.dumps(db_message))
                elif message.message_type == MessageType.ADD_REACTION:
                    db_message = {
                        "username": message.username,
                        "message_id": message.content.id,
                        "reaction": parse_reaction(message.content.reaction),
                    }
                    response = db_client.table(
                        "reactions").insert(db_message).execute()
                    # Add coins to the client
                    # Get number of coins
                    response = (
                        db_client.table("users")
                        .select("anti_social_credit")
                        .eq("username", message.username)
                        .execute()
                    )
                    credits = response.data[0]["anti_social_credit"]
                    credits += 100
                    db_message = {"username"}
                    response = (
                        db_client.table("users")
                        .update({"anti_social_credit": credits})
                        .eq("username", message.username)
                        .execute()
                    )

                    # TODO: Forward this to clients
                elif message.message_type == MessageType.REMOVE_REACTION:
                    response = (
                        db_client.table("reactions")
                        .delete()
                        .eq("username", message.username)
                        .eq("message_id", message.content.id)
                        .eq("reaction", parse_reaction(message.content.reaction))
                        .execute()
                    )
                    # TODO: Forward this to clients
            else:
                raise HTTPException(
                    status_code=401, detail="Invalid token, please reauthenticate!"
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)


"""ANTI-SOCIAL CREDIT CODE RESIDES HEREIN"""


@app.get("/antiSocialCredit/{username}")
async def get_anti_social_credit(username: str):
    try:
        response = (
            db_client.table("users")
            .select("username, anti_social_credit")
            .eq("username", username)
            .execute()
        )

        if not response.data:
            raise HTTPException(status_code=404, detail="Invalid username")

        return response.data[0]

    except Exception as e:
        if e.status_code == 404:
            raise HTTPException(status_code=404, detail="Invalid username")
        else:
            print(f"Something went wrong: {e}")
            raise HTTPException(
                status_code=500, detail="Internal Server Error")


"""GRASS CODE IS HERE"""
# How to use:
# In the websocket loop, when you're iterating over all the users, just poll
#     the grass_status and return to user's front end their status


@app.get("/grass/{username}")
async def grass_status(username: str):
    if await user_exists(username):
        try:
            response = (
                db_client.table("grass")
                .select("username, x_coord, y_coord, grass_number, size, growth_rate")
                .eq("username", username)
                .execute()
            )
            return response.data[0]
        except Exception as e:
            if e.status_code == 404:
                raise HTTPException(status_code=404, detail="Invalid username")
            else:
                print(f"Something went wrong: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal Server Error")


# adds more grass, does not grow the grass that already exists
# number of grass to plant is based on AI assessment of cringe
#      or the number of reactions


@app.post("/grass/{username}/{num_grass_to_plant}")
async def plant_grass(username: str, num_grass_to_plant: int = 1):
    if await user_exists(username):
        try:
            response = response = (
                db_client.table("grass")
                .select("username, x_coord, y_coord, grass_number, size, growth_rate")
                .eq("username", username)
                .execute()
            )
            if not response.data:  # user has no grass records, start grass growth
                db_client.table("grass").insert(
                    {
                        "username": username,
                        "x_coord": (int)(random.random() * 100),
                        "y_coord": (int)(random.random() * 100),
                        "grass_number": 1,
                        # range of 5-20
                        "size": (int)(random.random() * 15) + 5,
                        # 100 +/- 20
                        "growth_rate": (int)(random.random() * 20) + 100,
                    }
                ).execute()
            else:
                db_client.table("grass").update(
                    {
                        "grass_number": response.data[0]["grass_number"]
                        + (int)(num_grass_to_plant)
                    }
                ).eq("username", username).execute()
        except Exception as e:
            print(f"Something went wrong: {e}")
            raise HTTPException(
                status_code=500, detail="Internal Server Error")


# grow the grass that already exists, does not add more grass


@app.delete("/grass/{username}")
async def remove_grass(username: str):
    if await user_exists(username):
        try:
            response = (
                db_client.table("grass")
                .select("username, x_coord, y_coord, grass_number, size, growth_rate")
                .eq("username", username)
                .execute()
            )
            if not response.data:  # user has no grass records, start grass growth
                plant_grass(username)
            else:
                db_client.table("grass").update(
                    {"grass_number": response.data[0]["grass_number"] - 1}
                ).eq("username", username).execute()
        except Exception as e:
            print(f"Something went wrong: {e}")
            raise HTTPException(
                status_code=500, detail="Internal Server Error")


@app.patch("/grass/{username}")
async def grow_grass(username: str):
    if await user_exists(username):
        try:
            response = (
                db_client.table("grass")
                .select("username, x_coord, y_coord, grass_number, size, growth_rate")
                .eq("username", username)
                .execute()
            )
            if not response.data:  # user has no grass records, start grass growth
                plant_grass(username)
            else:
                db_client.table("grass").update(
                    {
                        "size": response.data[0]["size"]
                        * (response.data[0]["growth_rate"])
                    }
                ).eq("username", username).execute()
        except Exception as e:
            print(f"Something went wrong: {e}")
            raise HTTPException(
                status_code=500, detail="Invternal Server Error")


# fertilize the grass (increase growth rate)
# can make the fertilizer extra strong if their messages were extra cringe
async def fertilize_grass(username: str, fertilizer_strength: int = 20):
    if await user_exists(username):
        try:
            response = (
                db_client.table("grass")
                .select("username, x_coord, y_coord, grass_number, size, growth_rate")
                .eq("username", username)
                .execute()
            )
            if not response.data:  # user has no grass records, start grass growth
                plant_grass(username)
            else:
                db_client.table("grass").update(
                    {
                        "size": response.data[0]["size"]
                        * (response.data[0]["growth_rate"]),
                        "growth_rate": response.data[0]["growth_rate"]
                        + (int)(fertilizer_strength),
                    }
                ).eq("username", username).execute()
        except Exception as e:
            print(f"Something went wrong: {e}")
            raise HTTPException(
                status_code=500, detail="Internal Server Error")


async def user_exists(username: str):
    try:
        response = (
            db_client.table("users")
            .select("username")
            .eq("username", username)
            .execute()
        )
        if len(response.data) != 1:
            raise HTTPException(status_code=404, detail="Invalid username")
        else:
            return True

    except Exception as e:
        print(f"Something went wrong: {e}")
        if e.status_code == 404:
            raise HTTPException(status_code=404, detail="Invalid username")
        else:
            raise HTTPException(
                status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    uvicorn.run("example:app", host="127.0.0.1", port=8000, reload=True)
