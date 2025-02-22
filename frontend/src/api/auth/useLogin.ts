import supabase from "utils/supabase";

interface ILoginPostData {
	username: string;
	password: string;
}

export const login = async (postData: ILoginPostData) => {
	const { data, error } = await supabase.auth.signInWithPassword({
		email: postData.username,
		password: postData.password
	});

	if (error) return { data: null, error, isError: true };

	// TODO: Save the token to local storage if the data is true

	return { data, error: null, isError: false };
}
