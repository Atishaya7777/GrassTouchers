import { Navigate, Outlet } from "react-router";
import { ProtectedRouteProps } from "./ProtectedRoute.types";
import { routePaths } from "global/routePaths";

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
	isAuthenticated,
	redirectPath = routePaths.auth.login,
}) => {
	return isAuthenticated ? <Outlet /> : <Navigate to={redirectPath} replace />;
};

export default ProtectedRoute;
