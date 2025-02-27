import * as localStorage from './storage';

type TTokenObj = {
	accessToken: string;
};

const ACCESS_TOKEN = 'access_token';

type TReturnTypes = {
	setToken: (tokenObj: TTokenObj) => void;
	getAccessToken: () => string | null;
	clearToken: () => void;
};

const token = (): TReturnTypes => {
	function _setToken(tokenObj: TTokenObj) {
		localStorage.set(ACCESS_TOKEN, tokenObj.accessToken);
	}

	function _getAccessToken() {
		return localStorage.get(ACCESS_TOKEN) === 'undefined' ? undefined : localStorage.get(ACCESS_TOKEN);
	}

	function _clearToken() {
		localStorage.remove(ACCESS_TOKEN);
	}

	return {
		setToken: _setToken,
		getAccessToken: _getAccessToken,
		clearToken: _clearToken,
	};
};

export default token();
