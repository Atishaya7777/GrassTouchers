import { useRegister } from 'api/auth';
import { useUserStore } from 'store/useUserStore';
import { useState, useEffect } from "react";
import { Button, Input } from "components";
import { useForm } from "react-hook-form";
import { useNavigate } from "react-router";
import tokenService from 'utils/token';
import { routePaths } from "global/routePaths";
import useMediaQuery from "hooks/useMediaQuery";

interface RegisterFormData {
	username: string;
	password: string;
}

const Register = () => {
	const [showPassword, setShowPassword] = useState<boolean>(false);

	const navigate = useNavigate();
	const { register, handleSubmit, formState: { errors } } = useForm<RegisterFormData>();

	const addUserDetails = useUserStore(state => state.addUserDetails);

	const { mutate: registerMutate } = useRegister();

	const onSubmit = (data: RegisterFormData) => {
		addUserDetails(data);
		registerMutate(data);
	}

	useEffect(() => {
		if (tokenService.getAccessToken()) {
			navigate(routePaths.globalChat);
		}
	}, [])


	const isMobile = useMediaQuery("(max-width: 768px)");

	const renderPasswordEyes = () => {
		return (
			<button className="w-fit cursor-pointer" type="button" onClick={() => setShowPassword(!showPassword)}>
				{showPassword ? <p className="text-sm">Hide</p> : <p className="text-sm">Show</p>}
			</button>
		)
	}
	return (
		<div className="register-container">
			<div className="bg-gray-800/80 text-xl p-6 rounded-lg w-[75%] text-gray-100">
				<h1 className="text-3xl mb-5">{isMobile ? "Register" : "Register to avoid touching grass!"}</h1>
				<form className="flex flex-col justify-center text-2xl" onSubmit={handleSubmit(onSubmit)}>
					<div className="flex flex-col md:gap-15 mb-2 md:flex-row">
						<label>User:</label>
						<div className="w-full ml-0 md:ml-15">
							<Input className="font-bold" type="text" id="username" {...register("username", { required: true, maxLength: 15 })} />
							{errors.username && <span className="ml-4 text-lg text-red-700">This field is required</span>}
						</div>
					</div>
					<div className="flex flex-col md:gap-15 mb-5 md:flex-row">
						<label>Password:</label>
						<div className="w-full">
							<Input rightIcon={renderPasswordEyes()} type={showPassword ? "text" : "password"} id="password" {...register("password", { required: true })} />
							{errors.password && <span className="ml-4 text-lg text-red-700">This field is required</span>}
						</div>
					</div>
					<div className="flex justify-center">
						<Button type="submit">Login</Button>
					</div>
				</form>
			</div>
		</div>
	)
}

export default Register
