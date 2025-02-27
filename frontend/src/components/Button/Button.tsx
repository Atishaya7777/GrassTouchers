import { ButtonProps } from "./Button.types";
import React from "react";

const Button: React.FC<ButtonProps> = ({ className, isActive = false, ...rest }) => {
	return (
		<button className={["cursor-pointer p-2 w-fit  border-black border-3 hover:bg-gray-400/75", isActive ? "bg-black" : "bg-gray-400/50", className].join(" ")} {...rest}>{rest.children}</button>
	)
}

export default Button
