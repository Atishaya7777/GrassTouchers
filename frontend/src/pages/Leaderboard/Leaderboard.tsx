import { useState } from "react";
import { useFetchLeaderboard } from "api/leaderboard/useLeaderboard";
import arrow from 'assets/arrow.svg';
import { Button, FullPageLoader } from "components";
import { useNavigate } from "react-router";
import { routePaths } from "global/routePaths";

type TQueryState = 'messages' | 'credits';

const Leaderboard = () => {
  const navigate = useNavigate();
  const [queryState, setQueryState] = useState<TQueryState>('messages');

  const { data, isLoading } = useFetchLeaderboard({
    query: queryState
  });

  function formatNumber(num: number) {
    if (num < 1000) return num.toString();

    const suffixes = ["", "K", "M", "B", "T"]; // Up to Trillion
    let magnitude = Math.floor(Math.log10(num) / 3);

    if (magnitude >= suffixes.length) magnitude = suffixes.length - 1; // Cap at the highest suffix

    const truncated = (num / Math.pow(10, magnitude * 3)).toFixed(1);

    return `${truncated.replace(/\.0$/, '')}${suffixes[magnitude]}`;
  }

  const transformData = (data: any) => {
    return Object.keys(data).map((key) => {
      return {
        name: key,
        socialCredit: data[key]
      }
    }).sort((a, b) => b.socialCredit - a.socialCredit).slice(0, 10);
  }

  const isMessages = queryState === 'messages';

  const handleGoBack = () => {
    navigate(routePaths.globalChat);
  }

  if (isLoading) return <FullPageLoader />;

  return (
    <div className="container">
      <div className="p-2">
        <Button onClick={handleGoBack}>
          <img src={arrow} alt="arrow" />
        </Button>
        <div className="flex items-center justify-center flex-col mt-16">
          <h1 className="text-4xl text-leaderboard-header leader-board-header">LEADERBOARD</h1>
          <h3 className="text-2xl text-white mb-4">Most <span className={isMessages ? "text-red-600" : "text-green-600"}>{isMessages ? "Messages" : "gbucks"}</span></h3>
          <div className="flex gap-4">
            <Button className="text-white" isActive={isMessages} onClick={() => setQueryState('messages')}>Messages</Button>
            <Button className="text-white" isActive={!isMessages} onClick={() => setQueryState('credits')}>gBucks</Button>
          </div>

          <div className="flex flex-col gap-3 w-2/3 mt-10">
            {transformData(data?.payload || {})?.map((user, index) => (
              <div key={index} className="flex items-center justify-between w-full bg-gray-300 p-2 text-2xl">
                <div className="bg-gray-300 flex gap-2 items-center">
                  <div className="bg-black h-10 w-10 p-2 flex items-center justify-center">
                    <p className="text-gray-300 text-2xl">{index + 1}</p>
                  </div>
                  <p className="text-black text-2xl truncate max-w-full">{user.name}</p>
                </div>
                <p className="text-black">{formatNumber(user.socialCredit)}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Leaderboard;
