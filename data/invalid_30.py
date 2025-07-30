
def async_function():
    async def inner():
        await asyncio.sleep(1)
        return "done"
    
    return inner()
        