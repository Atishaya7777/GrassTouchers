import React from 'react';

import { isEmptyObject } from 'utils/helpers';

interface IErrorMapper {
	errorObj: any;
	fallbackText: any;
}

const ErrorMapper: React.FC<IErrorMapper> = ({ errorObj, fallbackText }) => {
	if (isEmptyObject(errorObj)) return <div>{fallbackText}</div>;

	return (
		<div>
			{Object.values(errorObj).map((errorValue: any, index: number) => {
				return <div key={index}>{errorValue[0]}</div>;
			})}
		</div>
	);
};

export default ErrorMapper;
