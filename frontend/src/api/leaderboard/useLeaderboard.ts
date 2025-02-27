import { useQuery } from 'react-query';

import { endpoints } from 'global/endpoints';
import http from 'utils/https';
import { stringifyQuery } from 'utils/query';

export interface IFetchLeaderboardQuery {
}

export const fetchLeaderboard = (searchQuery: IFetchLeaderboardQuery) => {
	const stringifiedQuery = stringifyQuery(searchQuery);
	const newEndpoints = endpoints.leaderboard + '?' + stringifiedQuery;
	return http().get(newEndpoints);
};

export function useFetchLeaderboard(searchQuery: IFetchLeaderboardQuery, options?: any) {
	return useQuery(['fetchLeaderboard', searchQuery], () => fetchLeaderboard(searchQuery), options);
}
