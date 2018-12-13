public class Test{
    public int f(int[] arr, int target) {
        for (int i = 0; i < arr.length; ++i){
            if (arr[i] == target){
                return i;
            }
        }
        return -1;
    }
}