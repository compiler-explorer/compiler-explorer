public static class Program
{
    public static int TestFunction(int[] input)
    {
        var sum = 0;
        foreach (var i in input)
        {
            sum += i;
        }
        return sum;
    }
}
