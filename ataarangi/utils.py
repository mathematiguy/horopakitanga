from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from os import cpu_count


def multicore_apply(array, func, n_jobs=cpu_count() - 1, use_kwargs=False, front_num=0):
    """
    A parallel version of the map function with a progress bar.

    This function applies a specified function `func` to each element of the
    provided `array` using multiple cores. It optionally processes the first few
    elements serially, which is useful for catching bugs early in development.

    Args:
        array (array-like): An iterable (like a list) where each element is either
            a single argument to `func` or, if `use_kwargs` is True, a dictionary
            of keyword arguments for `func`.
        func (callable): The function to apply to each element of `array`. If
            `use_kwargs` is True, `func` should accept keyword arguments.
        n_jobs (int, optional): The number of worker processes to use. Defaults
            to the number of available CPU cores minus one.
        use_kwargs (bool, optional): If True, each element in `array` is expected
            to be a dictionary. These dictionaries are unpacked and passed to
            `func` as keyword arguments. If False, each element in `array` is passed
            to `func` as a single argument.
        front_num (int, optional): The number of elements at the beginning of `array`
            to process serially before processing in parallel. Useful for debugging.
            If set to a negative number, the function will process all elements serially.

    Returns:
        list: A list containing the results of applying `func` to each element
            of `array`, in the order they appeared in `array`.

    Example:
        def add_one(x):
            return x + 1

        # Example without keyword arguments
        results = multicore_apply([1, 2, 3], add_one)

        # Example with keyword arguments
        def add_numbers(a, b):
            return a + b

        inputs = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
        results_with_kwargs = multicore_apply(inputs, add_numbers, use_kwargs=True)
    """
    array = list(array)

    # Run without parallelism if front_num is less than zero
    if front_num < 0:
        return [func(**a) if use_kwargs else func(a) for a in tqdm(array)]

    # We run the first few iterations serially to catch bugs
    front = []
    if front_num > 0:
        front = [func(**a) if use_kwargs else func(a) for a in array[:front_num]]
    # If we set n_jobs to 1 or front_num covers the entire array, just run a list comprehension.
    if n_jobs == 1 or front_num >= len(array):
        return front + [
            func(**a) if use_kwargs else func(a) for a in tqdm(array[front_num:])
        ]
    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into func
        if use_kwargs:
            futures = [pool.submit(func, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(func, a) for a in array[front_num:]]
        # Collect results as they are completed
        results = [
            future.result()
            for future in tqdm(as_completed(futures), total=len(futures))
        ]
        return front + results


def split_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


def invert_dictionary(original_dict):
    inverted_dict = {}
    for key, value in original_dict.items():
        if value not in inverted_dict:
            inverted_dict[value] = [key]
        else:
            inverted_dict[value].append(key)
    return inverted_dict
