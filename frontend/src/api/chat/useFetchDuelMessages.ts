import { useQuery } from 'react-query';

import { endpoints } from 'global/endpoints';
import http from 'utils/https';
import { stringifyQuery } from 'utils/query';

export interface IFetchDuelMessagesQuery {
	// NOTE: Add your query params here
}

export const fetchDuelMessages = (searchQuery: IFetchDuelMessagesQuery) => {
	const stringifiedQuery = stringifyQuery(searchQuery);
	const newEndpoints = endpoints.chat.duel.messages + '?' + stringifiedQuery;
	return http().get(newEndpoints);
};

export function useFetchDuelMessages(searchQuery: IFetchDuelMessagesQuery, options?: any) {
	return useQuery(['fetchDuelMessages', searchQuery], () => fetchDuelMessages(searchQuery), options);
}
