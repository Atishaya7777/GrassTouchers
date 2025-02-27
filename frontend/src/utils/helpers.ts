/**
 * Check for empty object
 * @param {Object} value Object
 * @returns {Boolean}
 */
export function isEmptyObject(obj: { [key: string]: any }) {
	return Object.keys(obj).length === 0 && obj.constructor === Object;
}

/**
 * Check for empty object and possibly undefined
 * @param {Object} value Object
 * @returns {Boolean}
 */
export function isEmptyObjectWithNull(obj?: { [key: string]: any }) {
	// We need to use a juggling-check as !{property: value} will evaluate to false
	if (obj == null) return obj;
	return Object.keys(obj).length === 0 && obj.constructor === Object;
}

/**
 * Check for object with empty values
 * @param {Object} value Object
 * @returns {Boolean}
 */
export function isEmptyObjectValues(obj: { [key: string]: any }) {
	return !Object.values(obj).some(element => element);
}
