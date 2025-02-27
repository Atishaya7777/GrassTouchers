import { create } from 'zustand';
import { combine, persist } from 'zustand/middleware';

interface IUserInitialState {
	user: null | {
		username: string;
		password: string;
	}
}

const userInitialState: IUserInitialState = {
	user: null
}

const userStore = combine(userInitialState, set => ({
	addUserDetails: (userDetails: any) => set(prev => ({ ...prev, user: userDetails })),
	clearUserDetails: () => set(prev => ({ ...prev, user: null }))
}));

export const useUserStore = create(persist(userStore, { name: 'user' }));
