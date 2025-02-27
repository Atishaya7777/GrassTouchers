import tokenService from 'utils/token';

export type THeader = 'json' | 'multipart' | 'default';

/**
 * transform header based on headerType
 *
 * @param {THeader} headerType
 */
export const getHeader = (headerType: THeader) => {
	const access_token = tokenService.getAccessToken();
	let header: HeadersInit = {};

	switch (headerType) {
		case 'json':
			header = {
				'Content-Type': 'application/json',
				Authorization: access_token ? `Bearer ${access_token}` : '',
			};
			break;

		case 'multipart':
			header = {
				'Content-Type': 'multipart/form-data',
				Authorization: access_token ? `Bearer ${access_token}` : '',
			};
			break;

		case 'default':
			header = { 'Content-Type': 'application/json' };
			break;

		default:
			header = { 'Content-Type': 'application/json' };
	}

	return header;
};
