import { Button, Input, Navbar } from "components";
import { useState, useEffect } from "react";
import { useUserStore } from "store/useUserStore";
import tokenService from "utils/token";
import heart from "assets/heart.svg";
import fire from "assets/fire.svg";
import angy from "assets/angy.svg";
import clown from "assets/clown.svg";
import nerd from "assets/nerg.svg";
import rollingEyes from "assets/rolling_eyes.svg";
import thistbh from "assets/thistbh.svg";
import cross from "assets/x.svg";

interface IMessage {
  id: number;
  text: string;
  timestamp: number;
  username: string;
  reactions: { [emoji: string]: Array<string> };
}

const GlobalChat = () => {
  const [messageText, setMessageText] = useState<string>("");
  const [messages, setMessages] = useState<IMessage[]>([]);
  const [hoveredMessage, setHoveredMessage] = useState<number | null>(null);
  const [reactionPanelVisible, setReactionPanelVisible] = useState<
    number | null
  >(null);


  const [socket, setSocket] = useState<WebSocket | null>(null);

  const userData = useUserStore((state) => state.user);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setMessageText(e.target.value);
  };

  const handleSendClick = () => {
    if (userData != null) {
      const message: IMessage = {
        id: 0,
        text: messageText,
        timestamp: Date.now(),
        username: userData.username,
        reactions: {}
      };
      const authenticatedMessage = {
        message: message,
        token: tokenService.getAccessToken(),
      };
      setMessageText(""); // Clear input after sending
      socket?.send(JSON.stringify(authenticatedMessage));
    }
  };

  const handleFormSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    handleSendClick();
  };

  useEffect(() => {
    const newSocket = new WebSocket("ws://localhost:8000/ws");
    newSocket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      message.reactions = {};
      setMessages((prevMessages) => [...prevMessages, message]);
    });

    setSocket(newSocket);
    return () => newSocket.close();
  }, []);

  const reactionIcons = [
    { src: heart, alt: "heart" },
    { src: fire, alt: "fire" },
    { src: angy, alt: "angy" },
    { src: clown, alt: "clown" },
    { src: nerd, alt: "nerd" },
    { src: rollingEyes, alt: "rolling eyes" },
    { src: thistbh, alt: "this tbh" },
    { src: cross, alt: "cross" },
  ];

  const handleReactionClick = (msgId: number, reaction: string) => {
    const messageIndex = messages.findIndex((msg) => msg.id === msgId);

    if (messageIndex === -1) return;

    const newMessages = [...messages];
    const message = newMessages[messageIndex];
    const reactions = { ...message.reactions };

    if (!reactions[reaction]) {
      reactions[reaction] = [];
    }

    if (reactions[reaction].includes(userData?.username as string)) {
      reactions[reaction] = reactions[reaction].filter(
        (user) => user !== userData?.username
      );
    } else {
      reactions[reaction].push(userData?.username as any);
    }

    newMessages[messageIndex] = { ...message, reactions };
    setMessages(newMessages);

    // TODO: Integrate API to persist reactions
  };

  const getFirstLetters = (name: string) => {
    return name
      .split(" ")
      .map((n) => n[0])
      .join("");
  };

  const formattedMessages = messages.reduce((acc, msg) => {
    const lastMessage = acc[acc.length - 1];
    const isSameUser = lastMessage?.username === msg.username;

    if (isSameUser) {
      lastMessage.messages.push(msg);
    } else {
      acc.push({ username: msg.username, messages: [msg] });
    }

    return acc;
  }, [] as { username: string; messages: IMessage[] }[]);

  return (
    <div className='container'>
      <Navbar />
      {socket?.readyState !== WebSocket.OPEN && (
        <p>Connection not established</p>
      )}

      <div className='mb-10' onMouseLeave={() => setHoveredMessage(null)}>
        {formattedMessages.length > 0 ? (
          formattedMessages.map((blockMsg, index) => {
            return (
              <div key={index} className='flex gap-3 p-2'>
                {/* NOTE: Avatar  section.*/}
                <div className='rounded-full h-12 w-12 flex items-center justify-center text-3xl capitalize bg-profile-bg text-gray-300'>
                  {getFirstLetters(blockMsg.username)}
                </div>

                {/* NOTE: Message section. */}
                <div className='flex flex-col w-full'>
                  <p className='text-sm text-purple-700 font-bold bg-input-bg border-black border-3 leading-none w-fit p-0.5 input-shadow'>
                    {blockMsg.username}
                  </p>
                  {blockMsg.messages.length > 0 &&
                    blockMsg.messages.map((msg, index) => (
                      <div className='relative' key={msg.id}>
                        <div
                          className='w-full hover:bg-profile-bg/75 p-0.25'
                          onMouseEnter={() => setHoveredMessage(msg.id)}
                        >
                          <p
                            key={index}
                            className='text-lg font-bold bg-input-bg border-black border-3 leading-none w-fit p-0.5 input-shadow'
                          >
                            {msg.text}
                          </p>
                        </div>
                        {/* NOTE: Reactions Below Message */}
                        <div className='flex'>
                          {msg.reactions &&
                            Object.keys(msg.reactions).length > 0 && (
                              <div className='flex gap-1 mt-2'>
                                {Object.entries(msg.reactions)
                                  .filter(([_, users]) => users.length > 0)
                                  .map(([reaction, users], i) => (
                                    <button
                                      key={i}
                                      className={`flex items-center space-x-1 transition-transform transform ${users.includes(userData?.username as any)
                                        ? "scale-110 text-profile-bg/80"
                                        : "5hover:scale-110"
                                        }`}
                                      onClick={() =>
                                        handleReactionClick(msg.id, reaction)
                                      }
                                    >
                                      <img
                                        src={
                                          reactionIcons.find(
                                            (r) => r.alt === reaction
                                          )?.src
                                        }
                                        alt={reaction}
                                        className='w-4 h-4 cursor-pointer'
                                      />
                                      {/* RECALL: Users is a set here. */}
                                      <span className='text-white text-sm'>
                                        {users.length}
                                      </span>
                                    </button>
                                  ))}
                              </div>
                            )}
                        </div>

                        {/* NOTE: Reaction Picker (Appears on hovering a message). */}
                        {(hoveredMessage === msg.id ||
                          reactionPanelVisible === msg.id) && (
                            <div
                              className='absolute top-4.5 left-0 mt-2 flex gap-2 bg-profile-bg/80 p-2 rounded-lg shadow-lg z-10'
                              onMouseEnter={() =>
                                setReactionPanelVisible(msg.id)
                              }
                              onMouseLeave={() => setReactionPanelVisible(null)}
                            >
                              {reactionIcons.map((icon, i) => (
                                <button
                                  key={i}
                                  className='transition-transform transform hover:scale-110'
                                  onClick={() =>
                                    handleReactionClick(msg.id, icon.alt)
                                  }
                                >
                                  <img
                                    src={icon.src}
                                    alt={icon.alt}
                                    className='w-5 h-5 cursor-pointer'
                                  />
                                </button>
                              ))}
                            </div>
                          )}
                      </div>
                    ))}
                </div>
              </div>
            );
          })
        ) : (
          <p>No messages yet</p>
        )}
      </div>

      {/* Input and Send button */}
      <form
        className='flex input-shadow fixed bottom-0 w-full bg-input-bg'
        onSubmit={handleFormSubmit}
      >
        <Input
          type='text'
          value={messageText}
          onChange={handleInputChange}
          className='pl-2'
        />
        <Button type='submit' onClick={handleSendClick}>
          Send
        </Button>
      </form>
    </div>
  );
};

export default GlobalChat;
