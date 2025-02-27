import { Component, ErrorInfo } from 'react';

interface IState {
	hasError: boolean;
}

class ErrorBoundary extends Component<any, IState> {
	constructor(props: {}) {
		super(props);
		this.state = { hasError: false };
	}

	static getDerivedStateFromError(): IState {
		return { hasError: true };
	}

	componentDidCatch(error: Error, info: ErrorInfo) {
		if (import.meta.env.DEV) {
			console.error('ErrorBoundary caught an error:', error, info);
		}
	}

	refreshPage = () => {
		window.location.reload();
	};

	render() {
		if (this.state.hasError) {
			return (
				<div className="flex justify-center items-center h-[60vh] text-center" role="alert">
					<div>
						<h2 className="text-2xl font-bold">Something went wrong</h2>
						<p className="my-2">Oops! Looks like there was an error. We're on it.</p>
						<button
							onClick={this.refreshPage}
							className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
						>
							Refresh Page
						</button>
					</div>
				</div>
			);
		}
		return this.props.children;
	}
}

export default ErrorBoundary;
