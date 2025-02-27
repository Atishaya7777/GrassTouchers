import { useUserStore } from "store/useUserStore";
import tokenService from 'utils/token';

const useAuthenticated = () => {
  const user = useUserStore(state => state.user);

  const isAuthenticated = !!user?.username && !!tokenService.getAccessToken();

  return { isAuthenticated };
}

export default useAuthenticated;
