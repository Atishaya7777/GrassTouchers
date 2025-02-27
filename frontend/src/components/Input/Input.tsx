import React from "react"
import { InputProps } from "./Input.types"

const Input: React.FC<InputProps> = ({ className, value, onChange, rightIcon, ...rest }) => {
	if (rightIcon) {
		return (
			<div className="relative w-full">
				<input className={[className, 'border-black border-3 w-full hover:bg-gray-300/50'].join(" ")} value={value} onChange={onChange} {...rest} />
				<div className="absolute top-0 right-4 w-fit">
					{rightIcon}
				</div>
			</div>
		)
	}

	return (
		<input className={[className, 'border-black border-3 w-full hover:bg-gray-300/50'].join(" ")} value={value} onChange={onChange} {...rest} />
	)
}

export default Input;
