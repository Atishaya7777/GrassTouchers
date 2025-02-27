import axios, { AxiosError, AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { QueryClient } from 'react-query';

import { getHeader, THeader } from 'utils/header';
import * as localStorage from 'utils/storage';
import tokenService from 'utils/token';

const queryClient = new QueryClient();

const logout = () => {
	localStorage.clear();
	queryClient.clear();
	if (tokenService.getAccessToken()) {
		window.location.reload();
	}
};

// Default API will be your root
const API_ROOT = import.meta.env.VITE_API_ENDPOINT || '';
const TIMEOUT = 180000;

// Report Polling interval in milliseconds
const REPORT_POLL_INTERVAL = parseInt(import.meta.env.VITE_REPORT_POLL_INTERVAL) || 3000;

/**
 * API call utility function
 *
 * @param {THeader} headerType
 * @param {string} baseURL
 * @param {number} timeout
 * @returns
 */

const http = (headerType: THeader = 'json', baseURL: string = API_ROOT, timeout: number = TIMEOUT) => {
	const headers = getHeader(headerType);

	const client: AxiosInstance = axios.create({
		baseURL,
		timeout,
		headers,
	});

	// Intercept response object and handleSuccess and Error Object
	client.interceptors.response.use(handleSuccess, handleError);

	function handleSuccess(response: AxiosResponse) {
		return response;
	}

	/** Intercept any unauthorized request.
	 * status 401 means either accessToken is expired or invalid
	 * dispatch logout action accordingly **/
	function handleError(error: AxiosError) {
		const status = error?.response?.status;
		if (status === 401) {
			logout();
		}
		if (error?.response?.status !== 500) {
			if (error?.response?.data) {
				return Promise.reject(error?.response?.data);
			} else {
				return Promise.reject(error);
			}
		} else {
			return Promise.reject(error?.response?.data);
		}
	}

	function get(path: string, config?: AxiosRequestConfig) {
		return client.get(path, config).then(response => response.data);
	}

	function post(path: string, payload: any, config?: AxiosRequestConfig) {
		return client.post(path, payload, config).then(response => response.data);
	}

	function put(path: string, payload: any, config?: AxiosRequestConfig) {
		return client.put(path, payload, config).then(response => response.data);
	}

	function patch(path: string, payload: any, config?: AxiosRequestConfig) {
		return client.patch(path, payload, config).then(response => response.data);
	}

	function _delete(path: string, data?: any, config?: AxiosRequestConfig) {
		if (data) {
			return client.delete(path, { data: data }).then(response => response.data);
		}
		return client.delete(path, config).then(response => response.data);
	}

	// Long polling post request: used with report processing
	function lpost(path: string, payload: any, config?: AxiosRequestConfig) {
		return client.post(path, payload, config).then(response => {
			if (response.status === 200) {
				return new Promise((resolve, reject) => {
					const processUrl = response?.request?.responseURL;
					const interval = setInterval(async () => {
						try {
							const responseInterval = await client.get(processUrl, config);
							if (responseInterval.status !== 200) {
								clearInterval(interval);
								reject(responseInterval);
							}

							const processResponse = responseInterval.data?.payload;
							if (processResponse.report_status === 'failed') {
								throw 'Operation failed';
							} else if (processResponse.report_status === 'completed') {
								clearInterval(interval);
								resolve(responseInterval.data);
							}
						} catch (error) {
							clearInterval(interval);
							reject(error);
						}
					}, REPORT_POLL_INTERVAL);
				});
			}

			return response.data;
		});
	}

	return {
		get,
		post,
		put,
		patch,
		delete: _delete,
		lpost: lpost,
	};
};

export default http;
