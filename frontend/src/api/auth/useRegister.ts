import { endpoints } from "global/endpoints";
import { routePaths } from "global/routePaths";
import { useMutation } from "react-query";
import { useNavigate } from "react-router";
import { useUserStore } from "store/useUserStore";
import http from "utils/https";
import toast from "react-hot-toast";

interface IRegisterPostData {
	username: string;
	password: string;
}

export const register = async (postData: IRegisterPostData) => {
	return http().post(endpoints.auth.register, postData);
};

export const useRegister = () => {
	const navigate = useNavigate();
	const clearUserDetails = useUserStore(state => state.clearUserDetails);

	return useMutation(register, {
		onSuccess: () => {
			toast.success('Registration successful!');
			navigate(routePaths.auth.login);
		},
		onError: (error) => {
			clearUserDetails();
			console.error('% REGISTER ERROR', error);
		}
	})
}
