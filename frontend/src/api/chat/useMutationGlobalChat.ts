import http from "utils/https";
import { endpoints } from "global/endpoints";
import { useMutation } from "react-query";

export const sendMessage = (postData: IMessage) => {
	return http().post(endpoints.chat.global.messages, postData);
}

export const useSendMessage = () => {

	return useMutation(sendMessage, {
		onSuccess: ({ payload }) => {
      console.log("Chat was successful", payload);
		},
		onError: ({ error }) => {
			console.log("% CHAT ERROR", error);
			// TODO: Add a toaster to show an error occured
		}
	})
};
