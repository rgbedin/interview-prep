


function findPercentageAllocation(orders, warehouseNumber, n) {
    const allocations = Array(n).fill(undefined).map(() => []);

    for (let order of orders) {

        if (order[1] < order[2]) {
            allocations[0].push(order);
        }
        else {
            allocations[1].push(order);
        }
    }

    const counts = allocations.map((all) => all.reduce((acc, curr) => acc + curr[0], 0));
    const total = counts.reduce((acc, curr) => acc + curr, 0);

    return counts[warehouseNumber - 1] / total;
}

const res = findPercentageAllocation([[3, 3, 7]], 1);
console.debug(res);