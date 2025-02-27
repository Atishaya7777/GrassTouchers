// TODO: Add your API endpoints from fastapi here
export const endpoints = {
	auth: {
		login: '/login',
		register: '/register'
	},
	chat: {
		duel: {
			messages: 'duel/messages'
		},
		global: {
			messages: 'global/messages'
		},
	},
	money: {
		gbucks: '/antiSocialCredit/{username}'
	},
	grass: '/grass/{username}',
	leaderboard: '/leaderboard'
}
