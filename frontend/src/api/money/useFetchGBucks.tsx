import { useQuery } from 'react-query';

import { endpoints } from 'global/endpoints';
import http from 'utils/https';

export interface IFetchGBucksQuery {
	username: string;
}

export const fetchGBucks = (searchQuery: IFetchGBucksQuery) => {
	const newEndpoints = endpoints.money.gbucks.replace('{username}', searchQuery.username);
	return http().get(newEndpoints);
};

export function useFetchGBucks(searchQuery: IFetchGBucksQuery, options?: any) {
	return useQuery(['fetchGBucks ', searchQuery], () => fetchGBucks(searchQuery), options);
}
