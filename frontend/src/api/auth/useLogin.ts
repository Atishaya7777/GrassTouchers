import { endpoints } from "global/endpoints";
import http from "utils/https";
import { useMutation } from "react-query";
import { useNavigate } from "react-router";
import { routePaths } from "global/routePaths";
import { useUserStore } from "store/useUserStore";
import token from "utils/token";
import toast from "react-hot-toast";

interface ILoginPostData {
  username: string;
  password: string;
}

export const login = (postData: ILoginPostData) => {
  return http().post(endpoints.auth.login, postData);
};

export const useLogin = () => {
  const addUserDetails = useUserStore((state) => state.addUserDetails);
  const navigate = useNavigate();

  return useMutation(login, {
    onSuccess: (data) => {
      token.setToken({ accessToken: data.data.token });
      toast.success("Login successful");
      addUserDetails(data.data);
      navigate(routePaths.globalChat);
    },
    onError: (error) => {
      console.error("% LOGIN ERROR", error);
      toast.error((error as any)?.detail || "An error occurred. Please try again.");
    },
  });
};
