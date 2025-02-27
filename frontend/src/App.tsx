import { ErrorBoundary, ErrorMapper } from "components";
import AppRouter from "./AppRouter"
import { QueryClientProvider, QueryClient } from "react-query";
import { Toaster } from "react-hot-toast";

function App() {
  // NOTE: The console logs are for debugging purposes.
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: 1,
        refetchOnWindowFocus: false,
        onError: (errorData: any) => {
          console.log(errorData);
          let errorMessage: any = '';
          if (errorData?.errors) {
            errorMessage = <ErrorMapper errorObj={errorData.errors} fallbackText={errorData.message} />;
            console.error("% ERROR", errorMessage);
          } else {
            errorMessage = errorData?.message || errorData?.error?.message || 'Unable to fetch data';
            console.error("% ERROR", errorMessage);
          }
        },
      },
      mutations: {
        retry: 0,
        onError: (errorData: any) => {
          console.log(errorData);
          let errorMessage: any = '';
          if (errorData?.errors) {
            errorMessage = <ErrorMapper errorObj={errorData.errors} fallbackText={errorData.message} />;
            console.error("% ERROR", errorMessage);
          } else {
            errorMessage = errorData?.message || errorData?.error?.message || 'Unable to post data';
            console.error("% ERROR", errorMessage);
          }
        },
        onSuccess: (data: any) => {
          if (data?.message) {
            console.log("% SUCCESS", data?.message);
          }
        },
      },
    },
  });

  return (
    <>
      <ErrorBoundary>
        <QueryClientProvider client={queryClient}>
          <AppRouter />
          <Toaster />
        </QueryClientProvider>
      </ErrorBoundary>
    </>
  )
}

export default App
