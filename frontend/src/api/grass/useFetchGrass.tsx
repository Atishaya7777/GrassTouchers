import { useQuery } from 'react-query';

import { endpoints } from 'global/endpoints';
import http from 'utils/https';

export interface IFetchGrassQuery {
	username: string;
}

export const fetchGrass = (searchQuery: IFetchGrassQuery) => {
	const newEndpoints = endpoints.grass.replace('{username}', searchQuery.username);
	return http().get(newEndpoints);
};

export function useFetchGrass(searchQuery: IFetchGrassQuery, options?: any) {
	return useQuery(['fetchGrass', searchQuery], () => fetchGrass(searchQuery), options);
}
