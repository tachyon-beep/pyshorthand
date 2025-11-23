"""Common sorting and searching algorithms.

Role: Util
Collection of fundamental algorithms with complexity annotations.
"""

from functools import lru_cache


def binary_search(arr: list[int], target: int) -> int:
    """Binary search in sorted array.

    Time: O(log N)

    Args:
        arr: Sorted array
        target: Value to find

    Returns:
        Index of target, or -1 if not found
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


def quick_sort(arr: list[int]) -> list[int]:
    """Quick sort algorithm.

    Complexity: O(N log N) average, O(N²) worst case

    Args:
        arr: Array to sort

    Returns:
        Sorted array
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)


def merge_sort(arr: list[int]) -> list[int]:
    """Merge sort algorithm.

    Time: O(N log N)

    Args:
        arr: Array to sort

    Returns:
        Sorted array
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)


def merge(left: list[int], right: list[int]) -> list[int]:
    """Merge two sorted arrays.

    Complexity: O(N)

    Args:
        left: First sorted array
        right: Second sorted array

    Returns:
        Merged sorted array
    """
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result


def bubble_sort(arr: list[int]) -> list[int]:
    """Bubble sort algorithm.

    Runtime: O(N²)

    Args:
        arr: Array to sort

    Returns:
        Sorted array
    """
    n = len(arr)
    result = arr.copy()

    for i in range(n):
        swapped = False
        for j in range(n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
                swapped = True

        if not swapped:
            break

    return result


@lru_cache(maxsize=1000)
def fibonacci(n: int) -> int:
    """Compute nth Fibonacci number with memoization.

    Complexity: O(N) with caching, O(2^N) without

    Args:
        n: Index in Fibonacci sequence

    Returns:
        nth Fibonacci number
    """
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def matrix_multiply(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    """Multiply two matrices.

    Time: O(N³) for N×N matrices

    Args:
        a: First matrix [M×N]
        b: Second matrix [N×P]

    Returns:
        Product matrix [M×P]
    """
    m, n, p = len(a), len(a[0]), len(b[0])
    result = [[0 for _ in range(p)] for _ in range(m)]

    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += a[i][k] * b[k][j]

    return result


def find_kth_largest(arr: list[int], k: int) -> int:
    """Find kth largest element using quickselect.

    Complexity: O(N) average, O(N²) worst case

    Args:
        arr: Input array
        k: k value (1 = largest)

    Returns:
        kth largest element
    """

    def partition(left: int, right: int, pivot_index: int) -> int:
        pivot = arr[pivot_index]
        arr[pivot_index], arr[right] = arr[right], arr[pivot_index]

        store_index = left
        for i in range(left, right):
            if arr[i] < pivot:
                arr[store_index], arr[i] = arr[i], arr[store_index]
                store_index += 1

        arr[right], arr[store_index] = arr[store_index], arr[right]
        return store_index

    def select(left: int, right: int, k_smallest: int) -> int:
        if left == right:
            return arr[left]

        pivot_index = (left + right) // 2
        pivot_index = partition(left, right, pivot_index)

        if k_smallest == pivot_index:
            return arr[k_smallest]
        elif k_smallest < pivot_index:
            return select(left, pivot_index - 1, k_smallest)
        else:
            return select(pivot_index + 1, right, k_smallest)

    return select(0, len(arr) - 1, len(arr) - k)
