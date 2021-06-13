UseRandomSleep=$1

echo "param 1: $UseRandomSleep"

if ["$UseRandomSleep"=="yes"] 
then
    SleepTime=$(shuf -i 0-300 -n 1)
    echo "sleep for $SleepTime seconds"
    sleep $SleepTime
else
    echo 'no sleep; starting job....'
fi

pause

source env/bin/activate

python3 auto_tinder.py 

deactivate