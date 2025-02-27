type QueryData = {
	[key: string]: any;
};

/**
 * Encode query params from key value pair
 * @example
 * stringifyQuery({active: false})
 * => active=false
 *
 * @param  data - Query Key value pair Object
 * @returns {string}
 */
export function stringifyQuery(data: QueryData) {
	let query = '';
	for (const d in data) query += encodeURIComponent(d) + '=' + encodeURIComponent(data[d] || '') + '&';
	return query.slice(0, -1);
}

/**
 * Encode query params from key value pair
 * @example
 * stringifyQueryWithoutEncoding({active: false})
 * => active=false
 *
 * @param  data - Query Key value pair Object
 * @returns {string}
 */
export function stringifyQueryWithoutEncoding(data: QueryData) {
	let query = '';
	for (const d in data) query += d + '=' + (data[d] || '') + '&';
	return query.slice(0, -1);
}

/**
 * Encode query params from key value pair
 * @example
 * stringifyQuery({active: false})
 * => active=false
 *
 * @param  data - Query Key value pair Object
 * @returns {string}
 */
export function decodeQueryString(queryString: string) {
	return Object.fromEntries(new URLSearchParams(queryString));
}
